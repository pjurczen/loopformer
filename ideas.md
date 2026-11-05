# Ideas: more tokens, or more progress, per 45 minutes

The question (2026-11-05): the LoopFormer does the plain model's arithmetic per token (12 block applications for 3×4 or 6×2) with fewer weights, and its training step adds a second, shorter route, so in a fixed time budget it sees fewer tokens and trails the plain model (phase 1: 3.831 vs 3.693; B200: 3.346 vs 3.191; see README). Is there anything in the looped architecture that turns the shared block into an advantage in tokens per second, or in progress per second?

Ruled out for this experiment: running the short route less often, dropping its cross-entropy, replacing it by early-exit consistency, a fused attention kernel, and batch 256 (these change the training recipe or help the plain model equally).

Measured context: at batch 64 on an A100, a plain step is 302 ms and a LoopFormer step ~380 ms (`--compile whole`); the loss over the 32K vocabulary is ~20% of a plain step's FLOPs and runs twice per LoopFormer step; the B200 ran the plain model at ~15% of its peak and 4× the batch gave 1.25× the tokens per second, so the limit is memory-bound work, not matmul size.

## 1. Adaptive depth per token

Route each token out of the loop when it no longer needs more iterations; compute attention only among the tokens still active, and cache K/V only for them. Fewer loop applications per token on average means more tokens per 45 minutes; the lm_head and loss cost stays.

- Source: Mixture-of-Recursions, Bae et al. 2025, [arXiv 2507.10524](https://arxiv.org/abs/2507.10524): a light router assigns each token a recursion depth; across 135M–1.7B models, lower perplexity and higher throughput than vanilla and plain recursive models at equal training FLOPs.
- Related: Think-at-Hard, [arXiv 2511.08577](https://arxiv.org/abs/2511.08577) (iterate only on tokens a decider flags, skipping 93%); Mixture-of-Depths, [arXiv 2404.02258](https://arxiv.org/abs/2404.02258), the plain-model analogue with top-k tokens per layer and static shapes.
- Estimate for us: if half the tokens leave 6×2 after one loop, the loop work drops by a quarter; MoR reports more because the attention set shrinks too.
- Cost: a router per loop, a capacity rule for static shapes, a routing loss; the comparison with the plain model changes meaning (compute per token is no longer equal).

## 2. K/V reuse across loops

Later loops reuse the keys and values the first loop computed instead of projecting them again. Per token and block, the K and V projections are about 15% of the arithmetic, so each later loop saves ~15% of its cost and part of its attention memory. Small, cheap, loop-specific.

- Sources: the KV-sharing variant of Mixture-of-Recursions ([arXiv 2507.10524](https://arxiv.org/abs/2507.10524)); zero-shot KV-cache sharing in the recurrent-depth model, Geiping et al. 2025, [arXiv 2502.05171](https://arxiv.org/abs/2502.05171) §6.2, and in Ouro, [arXiv 2510.25741](https://arxiv.org/abs/2510.25741) §5.4.2 (both at inference); EPAS, [arXiv 2601.19089](https://arxiv.org/abs/2601.19089), shares QK activations across deep layers of plain LLaMA models during training for up to 11% training throughput at a similar loss curve.
- Our part: applying the sharing at training time inside the loop, with the conditioning (t, Δt) still varying per loop. Not in those papers as such.
- Estimate for us: for 6×2, the second loop's K/V projections, a few percent of the step.

## 3. Batching across loop positions in training

Shared weights let loop i of one sample set and loop j of another run in one kernel. In our step, the long route's first loop and the short route could run as one batch of 2B with per-sample conditioning: fewer, larger kernels.

- Source of the mechanism: continuous depth-wise batching, Relaxed Recursive Transformers, [arXiv 2410.20672](https://arxiv.org/abs/2410.20672) §2.4 (at inference, 2–3× with early exit). The training-time use is our extrapolation.
- Estimate for us: under 10%: at 32K tokens per matmul the kernels already fill an A100, and the B200 measurement says the limit is memory-bound work.

## Context from the scaling literature

- "How Much Is One Recurrence Worth?", [arXiv 2604.21106](https://arxiv.org/abs/2604.21106): looping a block r times is worth r^0.46 unique blocks at matched compute, so looping alone loses; truncated backprop lowers that to 0.38; hyper-connections raise it to 0.65, a lever for progress per compute rather than tokens per second.
- Loopie, [arXiv 2607.16051](https://arxiv.org/abs/2607.16051): looping plus MoE reported to beat compute-matched vanilla models at 6–20B parameters; the recipe is not in the abstract.
