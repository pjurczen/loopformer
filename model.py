"""The LoopFormer language model (arXiv 2602.11451).

The building blocks (Linear, the tied embedding, RMSNorm, RoPE, attention with QK-norm, SwiGLU) are copies of the
cs336 assignment components, so results compare with the plain transformer trained from them. New are the (t, step)
conditioning, the LoopFormer block and the looped forward pass.
"""
import math
from dataclasses import dataclass

import torch
from einops import einsum, rearrange
from jaxtyping import Bool, Float
from torch._higher_order_ops import while_loop

COMPILE_SCOPES = ("blocks", "whole", "loop", "none")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    loop_blocks: int  # k: the number of unique blocks, applied once per loop
    loop_max: int  # L: the loops of the long trajectory; the model needs it only to size the per-loop LoRA table
    d_model: int
    num_heads: int
    d_ff: int | None = None
    eps: float = 1e-5
    rope_theta: float = 10_000.0
    conditioning: str = "adaln"  # how a block learns where it is in the loop: adaln (scales + gates), gate, additive
    loop_lora_rank: int = 0  # per-loop low-rank deltas on the shared matrices; 0 turns them off

    def __post_init__(self):
        if self.conditioning not in CONDITIONINGS:
            raise ValueError(f"conditioning must be one of {CONDITIONINGS}, got {self.conditioning!r}")


class LoopLoRA(torch.nn.Module):
    """Per-loop low-rank deltas for one shared matrix (Relaxed Recursive Transformers): at loop position i the matrix
    acts as W + B_i A_i. B starts at zero, so a fresh model equals the fully shared one. `loop` holds the current
    position; the model sets it before each loop."""
    A: torch.nn.Parameter  # (loops, rank, in_features)
    B: torch.nn.Parameter  # (loops, out_features, rank)

    def __init__(self, in_features: int, out_features: int, rank: int, loops: int, device=None, dtype=None):
        super().__init__()
        bound = 1 / math.sqrt(in_features)
        self.A = torch.nn.parameter.Parameter(torch.empty(loops, rank, in_features, device=device, dtype=dtype).uniform_(-bound, bound))
        self.B = torch.nn.parameter.Parameter(torch.zeros(loops, out_features, rank, device=device, dtype=dtype))
        self.register_buffer("loop", torch.zeros((), dtype=torch.long, device=device), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.A.index_select(0, self.loop.view(1)).squeeze(0)
        b = self.B.index_select(0, self.loop.view(1)).squeeze(0)
        return einsum(einsum(x, a, "... d_in, r d_in -> ... r"), b, "... r, d_out r -> ... d_out")


class Linear(torch.nn.Module):
    weight: torch.nn.Parameter  # (out_features, in_features)
    lora: LoopLoRA | None

    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = torch.nn.parameter.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(tensor=self.weight, mean=0, std=std, a=-3 * std, b=3 * std)
        self.lora = None

    def attach_lora(self, rank: int, loops: int, device=None, dtype=None) -> None:
        out_features, in_features = self.weight.shape
        self.lora = LoopLoRA(in_features, out_features, rank, loops, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.linear rather than the original einsum: the same x @ W^T, but its weight gradient comes out in plain
        # row-major layout, which a torch while_loop requires of every tensor it carries through its backward pass.
        y = torch.nn.functional.linear(x, self.weight)
        return y if self.lora is None else y + self.lora(x)


def zero_(linear: torch.nn.Linear) -> torch.nn.Linear:
    """A linear layer that starts as the zero map, so what it feeds starts switched off (adaLN-Zero)."""
    torch.nn.init.zeros_(linear.weight)
    torch.nn.init.zeros_(linear.bias)
    return linear


class EmbeddingLookup(torch.nn.Module):
    """Token embeddings that share the lm_head's matrix, scaled by sqrt(d_model) (Vaswani et al., section 3.4)."""
    weight: torch.nn.Parameter
    scale_factor: float

    def __init__(self, weight: torch.nn.Parameter, scale_factor: float = 1.0):
        super().__init__()
        self.weight = weight
        self.scale_factor = scale_factor

    # token_ids: (batch_size, sequence_length)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids] * self.scale_factor


def rms_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    """x divided by its RMS over the last dimension, in float32. The caller scales the result (RMSNorm by its
    weight, the LoopFormer block by (1 + gamma) from the conditioning) and casts back to x's dtype."""
    x = x.to(torch.float32)
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms


class RMSNorm(torch.nn.Module):
    weight: torch.nn.Parameter  # (d_model, )
    eps: float

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.parameter.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (rms_normalize(x, self.eps) * self.weight).to(x.dtype)


class RotaryPositionalEmbedding(torch.nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        k = torch.arange(0, d_k, 2, device=device)
        positions = torch.arange(0, max_seq_len, 1, device=device).reshape(max_seq_len, 1)
        angles = positions / theta ** (k / d_k)
        self.register_buffer('sin', torch.sin(angles), persistent=False)
        self.register_buffer('cos', torch.cos(angles), persistent=False)

    # x (..., seq_len, d_k), token_positions (seq_len)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "... seq_len (half two) -> ... seq_len half two", two=2)
        a = x[..., 0]
        b = x[..., 1]
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        a_out = a * cos - b * sin
        b_out = a * sin + b * cos
        x_out = torch.stack([a_out, b_out], dim=-1)
        return rearrange(x_out, "... seq_len half two -> ... seq_len (half two)")


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_norm = x.subtract(x_max)
    return x_norm.exp() / x_norm.exp().sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
        Q: Float[torch.Tensor, " ... queries d_k"],
        K: Float[torch.Tensor, " ... keys d_k"],
        V: Float[torch.Tensor, " ... keys d_v"],
        mask: Bool[torch.Tensor, " ... queries keys"] | None
) -> Float[torch.Tensor, " ... seq_len d_v"]:
    d_k = Q.shape[-1]
    qk = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        qk = qk.masked_fill(~mask, float("-Inf"))
    qk = softmax(qk, dim=-1)
    return einsum(qk, V, "... queries keys, ... keys d_v -> ... queries d_v")


class CausalMultiHeadSelfAttention(torch.nn.Module):
    """One fused QKV projection, RMSNorm on each head's queries and keys (QK-norm), then RoPE."""
    device: torch.device | None
    d_k: int
    d_v: int
    d_model: int
    num_heads: int
    qkv_proj: Linear  # (hd_k + hd_k + hd_v, d_model)
    q_norm: RMSNorm
    k_norm: RMSNorm
    output_proj: Linear  # (d_model, hd_v)
    rope: RotaryPositionalEmbedding

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, eps: float = 1e-5,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = round(d_model / num_heads)
        self.d_v = self.d_k
        self.qkv_proj = Linear(d_model, num_heads * self.d_k + num_heads * self.d_k + num_heads * self.d_v, device=device, dtype=dtype)
        self.q_norm = RMSNorm(self.d_k, eps, device=device, dtype=dtype)
        self.k_norm = RMSNorm(self.d_k, eps, device=device, dtype=dtype)
        self.output_proj = Linear(num_heads * self.d_v, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)

    # x (..., seq_len, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        causal_mask = (torch.tril(torch.ones(seq_len, seq_len, device=self.device)) == 1)
        token_positions = torch.arange(seq_len, device=self.device)
        w_x = self.qkv_proj(x)  # (..., seq_len, hd_k + hd_k + hd_v)
        wq_x = w_x[..., :self.num_heads * self.d_k]
        wk_x = w_x[..., self.num_heads * self.d_k:2 * self.num_heads * self.d_k]
        wv_x = w_x[..., 2 * self.num_heads * self.d_k:]
        wq_x_i = rearrange(wq_x, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        wk_x_i = rearrange(wk_x, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        wq_x_i = self.q_norm(wq_x_i)
        wk_x_i = self.k_norm(wk_x_i)
        wv_x_i = rearrange(wv_x, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.num_heads)
        wq_x_i = self.rope(wq_x_i, token_positions)
        wk_x_i = self.rope(wk_x_i, token_positions)
        result = scaled_dot_product_attention(wq_x_i, wk_x_i, wv_x_i, causal_mask)  # (..., h, seq_len, d_v)
        result = rearrange(result, "... h seq_len d_v -> ... seq_len (h d_v)")
        return self.output_proj(result)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    w1: Linear  # (d_ff, d_model)
    w2: Linear  # (d_model, d_ff)
    w3: Linear  # (d_ff, d_model)

    def __init__(self, d_model: int, d_ff: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        if d_ff is None:
            d_ff = round(8 * d_model / 3 / 64) * 64
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class TimestepEmbedding(torch.nn.Module):
    """A scalar in [0, 1] to a d_model vector: fixed sine and cosine features at geometrically spaced frequencies,
    then a two-layer MLP. The same construction as the diffusion-model timestep embedding."""

    def __init__(self, d_model: int, frequency_size: int = 256, max_period: float = 10_000.0, zero_output: bool = False, device=None, dtype=None):
        super().__init__()
        half = frequency_size // 2
        frequencies = torch.exp(-math.log(max_period) * torch.arange(half, device=device, dtype=torch.float32) / half)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(frequency_size, d_model, bias=True, device=device, dtype=dtype),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, d_model, bias=True, device=device, dtype=dtype),
        )
        if zero_output:  # the embedding starts at zero, for a use where it is added rather than passed through a zero modulator
            zero_(self.mlp[-1])

    # value: a 0-d tensor, so a compiled caller sees a tensor input rather than a constant per distinct value
    def forward(self, value: torch.Tensor) -> torch.Tensor:  # (d_model, )
        angles = value * self.frequencies
        features = torch.cat([torch.cos(angles), torch.sin(angles)]).to(self.mlp[0].weight.dtype)
        return self.mlp(features)


class LoopFormerBlock(torch.nn.Module):
    """A pre-norm transformer block of the shared stack. Each subclass is one way for the block to learn where it is
    in the loop from the conditioning vector c (see BLOCKS). The forward passes stay separate on purpose: the
    ablation measures exactly the operations each variant adds."""
    conditions_input: bool = False  # True: c is added to the hidden state before the block instead of modulating it

    def __init__(self, d_model: int, num_heads: int, d_ff: int | None, eps: float, max_seq_len: int, theta: float,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = eps
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, eps=eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def attach_lora(self, rank: int, loops: int, device=None, dtype=None) -> None:
        """Per-loop low-rank deltas on the block's shared matrices (qkv, output, w1, w2, w3)."""
        for linear in (module for module in self.modules() if isinstance(module, Linear)):
            linear.attach_lora(rank, loops, device=device, dtype=dtype)

    def set_loop(self, position: int) -> None:
        # A new tensor each loop, never an in-place change: autograd (and a compiled block) keeps the index of each
        # loop for the backward pass, so the index a previous loop used must stay as it was.
        loras = [module for module in self.modules() if isinstance(module, LoopLoRA)]
        if loras:
            loop = torch.full((), position, dtype=torch.long, device=loras[0].loop.device)
            for lora in loras:
                lora.loop = loop

    def lora_parameters(self) -> list[torch.nn.Parameter]:
        return [p for module in self.modules() if isinstance(module, LoopLoRA) for p in module.parameters()]


class AdaLNBlock(LoopFormerBlock):
    """adaln: the norms have no weight of their own; c sets the scale after each norm, (1 + gamma), and gates each
    residual branch with alpha (adaLN-Zero, as in Diffusion Transformers). A fresh block passes its input through."""

    def __init__(self, d_model: int, *args, device=None, dtype=None):
        super().__init__(d_model, *args, device=device, dtype=dtype)
        self.modulator = zero_(torch.nn.Linear(d_model, 4 * d_model, bias=True, device=device, dtype=dtype))

    # x (batch, seq_len, d_model), c (d_model, )
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gamma1, gamma2, alpha1, alpha2 = self.modulator(silu(c)).chunk(4, dim=-1)
        x = x + alpha1 * self.attn(rms_normalize(x, self.eps).to(x.dtype) * (1 + gamma1))
        x = x + alpha2 * self.ffn(rms_normalize(x, self.eps).to(x.dtype) * (1 + gamma2))
        return x


class GateBlock(LoopFormerBlock):
    """gate: the plain block (norms with their learned weight) plus a gate alpha per branch from c. A fresh block
    passes its input through."""

    def __init__(self, d_model: int, *args, device=None, dtype=None):
        super().__init__(d_model, *args, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, self.eps, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, self.eps, device=device, dtype=dtype)
        self.modulator = zero_(torch.nn.Linear(d_model, 2 * d_model, bias=True, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        alpha1, alpha2 = self.modulator(silu(c)).chunk(2, dim=-1)
        x = x + alpha1 * self.attn(self.ln1(x))
        x = x + alpha2 * self.ffn(self.ln2(x))
        return x


class PlainBlock(LoopFormerBlock):
    """additive: the plain block. c is added to the hidden state before the block, not used here."""
    conditions_input = True

    def __init__(self, d_model: int, *args, device=None, dtype=None):
        super().__init__(d_model, *args, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, self.eps, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, self.eps, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


BLOCKS: dict[str, type[LoopFormerBlock]] = {"adaln": AdaLNBlock, "gate": GateBlock, "additive": PlainBlock}
CONDITIONINGS = tuple(BLOCKS)


class LoopFormerLM(torch.nn.Module):
    """loop_blocks shared blocks, applied once per step of a trajectory. Each step is conditioned on the time t reached
    so far and on its own size. The embeddings are tied with the lm_head."""
    token_embeddings: EmbeddingLookup
    time_embedding: TimestepEmbedding
    step_embedding: TimestepEmbedding
    blocks: torch.nn.ModuleList
    ln_final: RMSNorm
    lm_head: Linear

    def __init__(self, config: ModelConfig, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        block_class = BLOCKS[config.conditioning]
        # With an additive block, c is added to the hidden state, so it starts at zero: a fresh model is a plain looped model.
        self.conditions_input = block_class.conditions_input
        self.lm_head = Linear(config.d_model, config.vocab_size, device=device, dtype=dtype)
        self.token_embeddings = EmbeddingLookup(weight=self.lm_head.weight, scale_factor=math.sqrt(config.d_model))
        self.time_embedding = TimestepEmbedding(config.d_model, zero_output=self.conditions_input, device=device, dtype=dtype)
        self.step_embedding = TimestepEmbedding(config.d_model, zero_output=self.conditions_input, device=device, dtype=dtype)
        self.blocks = torch.nn.ModuleList(
            block_class(config.d_model, config.num_heads, config.d_ff, config.eps, config.context_length, config.rope_theta, device=device, dtype=dtype)
            for _ in range(config.loop_blocks))
        if config.loop_lora_rank:
            for block in self.blocks:
                block.attach_lora(config.loop_lora_rank, config.loop_max, device=device, dtype=dtype)
        self.ln_final = RMSNorm(config.d_model, config.eps, device=device, dtype=dtype)
        self.loop_max = config.loop_max
        self.loop_lora_rank = config.loop_lora_rank
        self._block_calls: list | None = None
        self._forward_call = None
        self._scope = "blocks"

    def compile_for_device(self, device: str, scope: str = "blocks") -> None:
        """Compiles the model for training. The inductor backend generates fused kernels on CUDA (where float32 matmuls
        may also use TF32 tensor cores) and Metal kernels on MPS (2.1× the eager step on an M2 Pro); on the CPU
        aot_eager only traces the graph. Checkpoints keep the plain modules' keys.

        scope "blocks" compiles each block on its own: a trajectory of any length reuses the one compiled block.
        scope "whole" compiles the whole forward pass with the loop unrolled: one graph per loop count, which lets the
        compiler decide across blocks what to save for the backward pass and what to recompute.
        scope "loop" compiles the whole forward pass with the loop as a torch while_loop: one graph for every loop
        count. Recompilations are logged, so a run shows how many graphs it built.
        scope "none" runs eagerly: readable stack traces, and the reference for checking a compiled run."""
        if scope not in COMPILE_SCOPES:
            raise ValueError(f"scope must be one of {COMPILE_SCOPES}, got {scope!r}")
        if scope == "loop" and self.loop_lora_rank:
            raise ValueError("per-loop LoRA sets its position on the blocks, which a while_loop body may not do; use scope 'blocks' or 'whole'")
        self._scope = scope
        if scope == "none":
            return
        if device.startswith("cuda"):
            torch.set_float32_matmul_precision("high")
        backend = "inductor" if device.startswith(("cuda", "mps")) else "aot_eager"
        if device.startswith("mps"):
            # torch 2.11's Metal code generator emits the bounds checks of indirect indexing with an undeclared index
            # variable (SyntaxError at compile time); eager mode checks the same indices anyway.
            torch._inductor.config.assert_indirect_indexing = False
        torch._logging.set_logs(recompiles=True)
        if scope == "blocks":
            self._block_calls = [torch.compile(block, backend=backend) for block in self.blocks]
        elif scope == "whole":
            torch._dynamo.config.cache_size_limit = 64
            self._forward_call = torch.compile(self._forward_unrolled, backend=backend)
        else:
            self._forward_call = torch.compile(self._forward_while, backend=backend, fullgraph=True)

    def forward(self, tokens: torch.Tensor, schedule: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the logits and the hidden state after the final norm, (batch, seq_len, vocab) and (batch, seq_len, d_model).

        The schedule enters the compiled part as a tensor, so its values are inputs of the graph, not constants. With
        scope "loop" it is padded to loop_max steps and the loop count travels as a tensor too, so every call has the
        same shapes and one graph serves every loop count."""
        device = tokens.device
        if self._scope == "loop":
            padded = torch.tensor(schedule + [0.0] * (self.loop_max - len(schedule)), device=device)
            return self._forward_call(tokens, padded, torch.tensor(len(schedule), device=device))
        steps = torch.tensor(schedule, device=device)
        return (self._forward_call or self._forward_unrolled)(tokens, steps)

    def _forward_unrolled(self, tokens: torch.Tensor, steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.token_embeddings(tokens)
        t = torch.zeros((), device=x.device)
        for position in range(steps.shape[0]):
            step = steps[position]
            x = self._loop_body(x, t, step, position)
            t = t + step
        return self._output(x)

    def _forward_while(self, tokens: torch.Tensor, steps: torch.Tensor, loops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        def cond_fn(position, t, x):
            return position < loops

        def body_fn(position, t, x):
            step = steps.index_select(0, position.view(1)).squeeze()
            return position + 1, t + step, self._loop_body(x, t, step)

        x = self.token_embeddings(tokens)
        start = (torch.zeros((), dtype=torch.long, device=x.device), torch.zeros((), device=x.device), x)
        _, _, x = while_loop(cond_fn, body_fn, start)
        return self._output(x)

    def _loop_body(self, x: torch.Tensor, t: torch.Tensor, step: torch.Tensor, position: int | None = None) -> torch.Tensor:
        """One loop: the k blocks, conditioned on where the loop is (t) and how far it goes (step)."""
        c = self.time_embedding(t) + self.step_embedding(step)
        if self.conditions_input:
            x = x + c
        if position is not None:
            for block in self.blocks:
                block.set_loop(position)
        for block in self._block_calls or self.blocks:
            x = block(x, c)
        return x

    def _output(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.ln_final(x)
        return self.lm_head(hidden), hidden

    def muon_parameters(self) -> list[torch.nn.Parameter]:
        """The 2-D matrices inside the blocks, the modulators included: the parameters Muon updates. Everything else
        (embeddings, conditioning MLPs, norm weights, biases) is for AdamW, and so are the per-loop LoRA factors:
        Muon's orthogonalized steps have a fixed size, which would move the zero-start B factors too hard."""
        lora = {id(p) for block in self.blocks for p in block.lora_parameters()}
        return [p for block in self.blocks for p in block.parameters() if p.ndim == 2 and id(p) not in lora]
