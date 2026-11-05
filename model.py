"""The LoopFormer language model (arXiv 2602.11451).

The building blocks (Linear, the tied embedding, RMSNorm, RoPE, attention with QK-norm, SwiGLU) are copies of the
cs336 assignment components, so results compare with the plain transformer trained from them. New are the (t, step)
conditioning, the gated block and the looped forward pass.
"""
import math
from dataclasses import dataclass

import torch
from einops import einsum, rearrange
from jaxtyping import Bool, Float


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    loop_blocks: int  # k: the number of unique blocks, applied once per loop
    d_model: int
    num_heads: int
    d_ff: int | None = None
    eps: float = 1e-5
    rope_theta: float = 10_000.0


class Linear(torch.nn.Module):
    weight: torch.nn.Parameter  # (out_features, in_features)

    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = torch.nn.parameter.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(tensor=self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


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


class RMSNorm(torch.nn.Module):
    weight: torch.nn.Parameter  # (d_model, )
    eps: float

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.parameter.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.to(torch.float32)  # the statistics in float32, whatever autocast runs the rest in
        rms = torch.sqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32 / rms * self.weight).to(x.dtype)


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

    def __init__(self, d_model: int, frequency_size: int = 256, max_period: float = 10_000.0, device=None, dtype=None):
        super().__init__()
        half = frequency_size // 2
        frequencies = torch.exp(-math.log(max_period) * torch.arange(half, device=device, dtype=torch.float32) / half)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(frequency_size, d_model, bias=True, device=device, dtype=dtype),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, d_model, bias=True, device=device, dtype=dtype),
        )

    # value: a 0-d tensor, so a compiled caller sees a tensor input rather than a constant per distinct value
    def forward(self, value: torch.Tensor) -> torch.Tensor:  # (d_model, )
        angles = value * self.frequencies
        features = torch.cat([torch.cos(angles), torch.sin(angles)]).to(self.mlp[0].weight.dtype)
        return self.mlp(features)


class LoopFormerBlock(torch.nn.Module):
    """A pre-norm transformer block of the shared stack, with a gate on each residual branch from the conditioning
    vector c: adaLN-Zero without its scales, which do no measurable work (docs/results.md, "Ablations"). The
    modulator starts at zero, so a fresh block passes its input through and every route gives the same output."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int | None, eps: float, max_seq_len: int, theta: float,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, eps=eps, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.modulator = zero_(torch.nn.Linear(d_model, 2 * d_model, bias=True, device=device, dtype=dtype))

    # x (batch, seq_len, d_model), c (d_model, )
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        alpha1, alpha2 = self.modulator(silu(c)).chunk(2, dim=-1)
        x = x + alpha1 * self.attn(self.ln1(x))
        x = x + alpha2 * self.ffn(self.ln2(x))
        return x


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
        self.lm_head = Linear(config.d_model, config.vocab_size, device=device, dtype=dtype)
        self.token_embeddings = EmbeddingLookup(weight=self.lm_head.weight, scale_factor=math.sqrt(config.d_model))
        self.time_embedding = TimestepEmbedding(config.d_model, device=device, dtype=dtype)
        self.step_embedding = TimestepEmbedding(config.d_model, device=device, dtype=dtype)
        self.blocks = torch.nn.ModuleList(
            LoopFormerBlock(config.d_model, config.num_heads, config.d_ff, config.eps, config.context_length, config.rope_theta, device=device, dtype=dtype)
            for _ in range(config.loop_blocks))
        self.ln_final = RMSNorm(config.d_model, config.eps, device=device, dtype=dtype)
        self._forward_call = None

    def compile_for_device(self, device: str) -> None:
        """Compiles the whole forward pass with the loop unrolled: one graph per loop count, so the compiler decides
        across blocks what to save for the backward pass and what to recompute (docs/results.md, "Compile scope").
        The inductor backend generates fused kernels on CUDA (where float32 matmuls may also use TF32 tensor cores)
        and Metal kernels on MPS (2.1× the eager step on an M2 Pro); on the CPU aot_eager only traces the graph.
        Recompilations are logged, so a run shows how many graphs it built. Checkpoints keep the plain modules' keys."""
        if device.startswith("cuda"):
            torch.set_float32_matmul_precision("high")
        backend = "inductor" if device.startswith(("cuda", "mps")) else "aot_eager"
        if device.startswith("mps"):
            # torch 2.11's Metal code generator emits the bounds checks of indirect indexing with an undeclared index
            # variable (SyntaxError at compile time); eager mode checks the same indices anyway.
            torch._inductor.config.assert_indirect_indexing = False
        torch._logging.set_logs(recompiles=True)
        torch._dynamo.config.cache_size_limit = 64
        self._forward_call = torch.compile(self._forward_steps, backend=backend)

    def forward(self, tokens: torch.Tensor, schedule: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the logits and the hidden state after the final norm, (batch, seq_len, vocab) and (batch, seq_len, d_model).

        The schedule enters the compiled part as a tensor, so its values are inputs of the graph, not constants: one
        graph per loop count rather than one per distinct schedule."""
        steps = torch.tensor(schedule, device=tokens.device)
        return (self._forward_call or self._forward_steps)(tokens, steps)

    def _forward_steps(self, tokens: torch.Tensor, steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.token_embeddings(tokens)
        t = torch.zeros((), device=x.device)
        for position in range(steps.shape[0]):
            step = steps[position]
            x = self._loop_body(x, t, step)
            t = t + step
        hidden = self.ln_final(x)
        return self.lm_head(hidden), hidden

    def _loop_body(self, x: torch.Tensor, t: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """One loop: the k blocks, conditioned on where the loop is (t) and how far it goes (step)."""
        c = self.time_embedding(t) + self.step_embedding(step)
        for block in self.blocks:
            x = block(x, c)
        return x

    def muon_parameters(self) -> list[torch.nn.Parameter]:
        """The 2-D matrices inside the blocks, the modulators included: the parameters Muon updates. Everything else
        (embeddings, conditioning MLPs, norm weights, biases) is for AdamW."""
        return [p for block in self.blocks for p in block.parameters() if p.ndim == 2]
