# Ideas: more tokens, or more progress, per 45 minutes

The question (2026-11-05): the LoopFormer does the plain model's arithmetic per token (12 block applications for 3×4 or 6×2) with fewer weights, and its training step adds a second, shorter route, so in a fixed time budget it sees fewer tokens and trails the plain model (phase 1: 3.831 vs 3.693; B200: 3.346 vs 3.191; see README). Is there anything in the looped architecture that turns the shared block into an advantage in tokens per second, or in progress per second?

Ruled out for this experiment: running the short route less often, dropping its cross-entropy, replacing it by early-exit consistency, a fused attention kernel, and batch 256 (these change the training recipe or help the plain model equally).

Measured context: at batch 64 on an A100, a plain step is 302 ms and a LoopFormer step ~380 ms (`--compile whole`); the loss over the 32K vocabulary is ~20% of a plain step's FLOPs and runs twice per LoopFormer step; the B200 ran the plain model at ~15% of its peak and 4× the batch gave 1.25× the tokens per second, so the limit is memory-bound work, not matmul size. Our data says ~0.13 nats of loss per doubling of tokens seen, so a 10% cheaper step is worth ~0.02 nats.

## How this list was made

Three research passes on 2026-11-05: one from first principles (what in our step could be cheaper), two through the literature (about 60 papers on looped, recurrent-depth and weight-shared transformers, read in their method and result sections). Every number below is either measured here (marked "ours") or quoted from a paper (arXiv id and section given). The quotes are the research agents' reading of the papers and were not re-checked by us. "Origin: ours" means the idea came up in this project's reasoning and we did not find it in the papers we read; it may exist elsewhere.

The short version:

- Nothing in a weight-tied model makes a training step cheaper by itself. A block's bf16 weights are 14 MB against 50 MB activation tensors, so reusing weights from cache cannot matter, and no paper reports a training kernel that profits from tied weights (the one hardware saving is the optimizer step over half the matrices, 1–2%). Tokens per second can only come from skipping work in later loops: ideas 1–5.
- Progress per token comes from small changes at the loop boundary, which the 2026 scaling studies converge on: ideas 6–10.
- Five compute-matched studies agree that a dense two-loop model is at best even with a plain model of the same compute (2604.21106, 2610.26030, 2609.01343, 2610.40325, 2607.16051). The realistic target is to close most of our 0.15-nat gap, not to invert it. Our choices of two loops, full backprop and a multiplicative per-loop signal are the ones those studies also land on.

## Summary

| # | Idea | Lever | Reported or estimated effect | Origin | Cost to try |
|---|------|-------|------------------------------|--------|-------------|
| 1 | Reuse the first loop's K/V in later loops | both | −0.027 nats at 2 loops (paper), saves K/V projections | LPT 2610.02383; training use with (t, Δt) is ours | medium |
| 2 | Reuse the first loop's attention pattern | tokens/s | −14–18% step for 3×4, less for 6×2 (our estimate) | ours | medium |
| 3 | Narrower later loops (Matryoshka) | tokens/s | −23% step for 3×4 (our estimate) | ours | medium |
| 4 | Start looping late (6×1 first, then 6×2) | tokens/s | first phase at the plain step time; most of the gain kept (papers) | ELR 2610.12582, UpLoop 2610.41720 | medium |
| 5 | Short-route loss on every 4th position | tokens/s | −9% step (our estimate); touches the ruled-out list | ours | small |
| 6 | Prelude–loop–coda (unique first and last blocks) | progress | −0.01 to −0.02 nats at matched compute (papers) | Geiping 2502.05171, SMELT 2609.01343 | small |
| 7 | Hyper-connections once per loop | progress | −0.02 to −0.04 nats vs vanilla looping, −5% throughput (papers) | Hyperloop 2604.21254, 2604.21106 §5.2 | medium |
| 8 | Gate sees the hidden state, per token | progress | −0.006 nats (per-token gate, paper); state gate: GRT beats MoR by 0.08 | ELR 2610.12582, GRT 2608.15062 | small |
| 9 | Stability: residual scaling, bounded injection | progress | recovers the plain model's lr and 120-step warmup (our loss: ~0.17 nats at step 1,000) | Loop-μP 2606.18524, DeepLoop 2607.13491, Parcae 2604.12946 | small |
| 10 | MoE in the shared FFN | progress | large gains in papers at 6–20B; 1.7× slower per batch at 244M | MoEUT 2405.16039, Loopie 2607.16051 | large |
| 11 | Adaptive depth per token | tokens/s | demoted: loses 0.08–0.16 nats at our scale (papers) | MoR 2507.10524 | large |
| 12 | Truncated backward through the last loops | tokens/s | −30% FLOPs, only for L ≥ 4; fails at L = 2 | Geiping, Parcae 2610.38422 | small |
| 13 | Batch loop positions together | tokens/s | under 10% (our estimate) | RRT 2410.20672; training use is ours | medium |
| 14 | Local shortcut consistency instead of a second route | tokens/s | +1 block instead of +L/2 per step; changes the recipe | Shortcut Models 2410.12557 | medium |

## A. Cheaper later loops (tokens per second)

### 1. Reuse the first loop's keys and values

The first loop computes K and V for every block as now. Later loops compute only their queries (plus, optionally, K/V for a short local window of 64 tokens) and attend to the first loop's K/V in one softmax. Per token and block, the K and V projections are about 15% of the arithmetic, so each later loop saves ~15% of its cost; for 6×2 that is the second loop only, a few percent of the step. The attention probability matrices do not shrink, so the 20 GB of attention memory stays.

- Sources: Loop Parallel Transformer, [arXiv 2610.02383](https://arxiv.org/abs/2610.02383) §3, Table 1: trained from scratch at 150M–1B, same parameters and loop count, <1% more FLOPs, perplexity 16.17 → 15.73 at 2 recursions (0.027 nats), 15.98 → 15.28 at 3, 16.00 → 14.91 at 5; vanilla looping stops improving beyond 3 recursions, LPT keeps improving; three extra seeds, p < 0.01. The paper's explanation: the shared cache is a direct gradient path into the first loop. Corroboration: Asymmetric Looped Transformer, [arXiv 2610.28489](https://arxiv.org/abs/2610.28489) §3.1 (one-pass encoder makes global K/V, a tied decoder loops; at 60M: full loop K=2 CE 3.043 at 0.232 s/step, AsymLoop K=3 2.968 at 0.216 s/step); MoR's KV-sharing variant, [arXiv 2507.10524](https://arxiv.org/abs/2507.10524) §4.3; [arXiv 2610.33360](https://arxiv.org/abs/2610.33360); at inference only: Geiping et al., [arXiv 2502.05171](https://arxiv.org/abs/2502.05171) §6.2, Ouro, [arXiv 2510.25741](https://arxiv.org/abs/2510.25741) §5.4.2; in plain models during training: EPAS, [arXiv 2601.19089](https://arxiv.org/abs/2601.19089), up to 11% training throughput at a similar loss curve.
- Origin: the papers. Our part is keeping the (t, Δt) conditioning per loop while K/V is shared.
- Cost: an attention mode in `GateBlock`/`AdaLNBlock` where loop 0 returns K, V and later loops take them as input; both compile scopes need the K/V carried across loops.
- Risk: at L = 2 the gain is the small end (0.027 nats); the one-loop short route is unaffected.

### 2. Reuse the first loop's attention pattern

Go one step further than idea 1: later loops reuse the softmax probabilities of the first loop, so they skip Q and K projections and the QKᵀ softmax, and only compute V, the weighted sum and the output projection. For 3×4 that is 14–18% of the step (our estimate; the attention memory also drops, because later loops store no probabilities).

- Sources for the premise that attention patterns are similar across layers: WISE, [arXiv 2609.27373](https://arxiv.org/abs/2609.27373) (sharing attention weights across layers of plain models); EPAS above. MoR §15 reports high cosine similarity of K/V across shared depths.
- Origin: ours (sharing the whole pattern across loops during training). Not found in the papers we read.
- Cost: as idea 1 plus carrying the probabilities, which conflicts with a fused attention kernel later.
- Risk: the second loop can no longer re-route attention, which may be where part of the loop's value lies; untested anywhere. Idea 1 is the safe half of this.

### 3. Narrower later loops (Matryoshka)

Later loops use a slice of the shared weights: half the heads and half of d_ff, so the second pass of 6×2 costs about half. The weights are the same tensors, sliced, so nothing new is learned; the model learns to make the first half of each matrix do most of the work. Our estimate: −23% step for 3×4, −32% combined with idea 2.

- Origin: ours. Related only in spirit to Matryoshka representation learning for embeddings; not found for loops.
- Cost: a width argument per loop in the block's forward; the lm_head and loss are unchanged.
- Risk: untested; the loop's second pass is a weaker model, and the elastic-depth evaluation changes meaning. Try after idea 1 if the time saving is needed.

### 4. Start looping late

Train 6×1 (six unique blocks, no loop, no short route) for the first part of the budget at the un-looped step time, then switch to 6×2. The first phase sees ~1.9× the tokens per second of the looped step. A stronger version copies blocks and their optimizer state in stages (UpLoop).

- Sources: ELR, [arXiv 2610.12582](https://arxiv.org/abs/2610.12582) §6.5 and [arXiv 2609.19107](https://arxiv.org/abs/2609.19107): switching to loops partway keeps most of the looped model's gain (84–97% in their tables) at a fraction of the extra cost. UpLoop, [arXiv 2610.41720](https://arxiv.org/abs/2610.41720) §4.1, §5.2.2: at 370M, vanilla 2.78 NLL at 1.0× FLOPs, loop from scratch 2.76 at 1.20×, UpLoop 2.74 at 1.0× FLOPs and tokens, by growing from L/4 layers at ~10%, 23%, 44% of the tokens and looping a middle block only in the last stage. Ouro §4.3.1 reports layer-duplication upcycling is "particularly smooth" for looped models.
- Origin: the papers.
- Cost: a stage schedule in `train.py`; one compile per stage (~80 s each); the loop_max and the short route switch on at the stage boundary.
- Risk: stage transitions inside 45 minutes need their own warmups; UpLoop's final model has the plain model's parameter count, so it answers "a looped model for free", not "does the looped model win".

### 5. Short-route loss on every 4th position

The short route's cross-entropy is computed on a strided subset of positions. The loss over the 32K vocabulary is ~20% of a plain step's FLOPs and runs twice per LoopFormer step; this removes three quarters of the second pass, ~9% of the step (our estimate). The consistency loss stays on every position.

- Origin: ours. Minor.
- Risk: this is a partial version of "dropping its cross-entropy", which is on the ruled-out list; noted for completeness.

## B. Better loops (progress per token)

### 6. Prelude–loop–coda

Keep the first and last one or two blocks unshared and loop only the middle. For 12 effective layers: 2 + 4×2 + 2 gives 8 unique blocks (~66M weights), close to the plain model's 85M. Our own phase-1 table already points this way (6×2 > 4×3 > 3×4 > 2×6: more unique blocks win). The arithmetic per token is unchanged; the positions that specialize most (next to the embedding and the head) get their own weights. The prelude is the same for both routes, so it runs once per step and both routes' gradients sum into it (our part).

- Sources: Geiping et al. 2502.05171 (the layout's origin); SMELT, [arXiv 2609.01343](https://arxiv.org/abs/2609.01343) §13.3, §13.5: at 200M MoE with per-token FLOPs, total parameters and KV matched, baseline 1.9445, middle half looped twice 1.9257, full stack looped 1.9322, 3 loops 1.9385, 4 loops 1.9360; Meta's sweep, [arXiv 2610.40325](https://arxiv.org/abs/2610.40325) (70M–1B): looping a middle block "gives a better return when training FLOPs are constrained", full-stack looping wins only at fixed parameters; [arXiv 2610.23036](https://arxiv.org/abs/2610.23036) (730M): the sandwich "consistently improves" over full looping; [arXiv 2610.26030](https://arxiv.org/abs/2610.26030) §6.3 (~600 runs): the compute-optimal (2 forward, 2 backward) model sits within 0.008 nats of the compute-optimal plain model with 27–32% fewer non-embedding parameters.
- Origin: the papers; computing the prelude once for both routes is ours.
- Cost: unshared `PlainBlock`s around the looped stack in `LoopFormerLM`; `trajectory.py` unchanged; `train.py` runs the prelude once and sums both losses before its backward (or retains its graph).
- Risk: gains in the papers are 0.01–0.02 nats at matched compute, not the full gap; 6×2 is already nearly the plain model with the top half tied.

### 7. Hyper-connections once per loop

Carry n = 2–4 copies ("lanes") of the residual stream across loops. Before each loop a learned per-loop n→1 mix forms the block input; after it a 1→n mix writes the output back; a diagonal sigmoid scales the carried lanes; the lanes are averaged before the final norm. A few dozen scalars per loop.

- Sources: Hyperloop, [arXiv 2604.21254](https://arxiv.org/abs/2604.21254) §3, §4.2: at 238M, 12.5B tokens, perplexity plain 14.65, vanilla looped 14.85, Hyperloop 14.40 (0.03 nats over vanilla looping, 0.017 over the plain model), training throughput 786K / 786K / 750K tok/s (−4.6%); at 1B 10.19 / 10.02 / 9.65; after 100B tokens the looped model's deficit to the plain one is 0.033 nats and Hyperloop's is zero. "How Much Is One Recurrence Worth?", [arXiv 2604.21106](https://arxiv.org/abs/2604.21106) §5.2, App C.2: hyper-connections raise the worth of a loop from r^0.46 to r^0.65 unique blocks, −0.036 nats at 40M; diagonal mixing worked best. Original hyper-connections: [arXiv 2409.19606](https://arxiv.org/abs/2409.19606); applied at every layer (mHC) they cost 35% of training throughput, so once per loop is the point.
- Origin: the papers.
- Cost: an [n, B, S, d] residual carried across loops; per-loop mixing weights indexed by the loop counter (a stacked tensor under `--compile loop`); memory n× the carried residual (~0.1 GB here).
- Risk: 2604.21106 §5.2 concedes hyper-connections may help the plain model too; interaction with the gates untested.

### 8. Gate sees the hidden state, per token

Our gate ablation showed the multiplicative per-loop signal is worth 0.023 nats and the (1 + γ) scales nothing. Two cheap upgrades: make the gate per token (a projection of the hidden state instead of a broadcast vector), and feed it the current hidden state and the prelude output as well as (t, Δt).

- Sources: ELR 2610.12582: per-token gates −0.006 nats at every depth; Gated Recurrent Transformer, [arXiv 2608.15062](https://arxiv.org/abs/2608.15062) §3.3–3.4, §4: an elementwise update gate conditioned on the hidden state, the prelude output and noise; at GPT-2-small scale under isoFLOPs, 3.145 ± 0.004 vs the dense 3.188 ± 0.056 (three seeds) with 36% of the parameters, and MoR and Poisson depth sampling trail GRT by 0.08–0.16 nats at every scale. Take only the gate: GRT's one-route depth sampling is on our ruled-out list.
- Origin: the papers.
- Cost: one extra input to `GateBlock`'s modulator; a per-token projection d→2d per block.
- Risk: small gains; the per-token modulation is a B×S×d tensor per block per loop (the thing our memory work removed), so measure memory.

### 9. Stability, to recover the learning rate the loop lost

The B200 run needed 500 warmup steps against the plain model's 120 and saw gradient bursts; at step 1,000 the looped model was 0.17 nats behind at equal wall-clock (README). Papers trace loop instability to the growth of the residual across revisits and to the input-injection parameters, and fix it by scaling the branch outputs or bounding those parameters.

- Sources: Loop-μP, [arXiv 2606.18524](https://arxiv.org/abs/2606.18524); DeepLoop, [arXiv 2607.13491](https://arxiv.org/abs/2607.13491): residual-scaling exponent 1/2 instead of DeepNorm's 1/4 for tied stacks, loss gains only once blocks are revisited; Parcae, [arXiv 2604.12946](https://arxiv.org/abs/2604.12946): constrains the spectral norm of the input-injection parameters, up to 6.3% lower perplexity than prior looped models; LOOM, [arXiv 2610.01153](https://arxiv.org/abs/2610.01153): scaled residual updates and the embedding re-injected every loop, stable to 9–12 loops; Ouro §4.3: recurrent models needed smaller learning rates and a doubled batch for stability, cut 8 steps to 4. Input injection on its own has mixed evidence (helps in Geiping/LOOM, not in Ouro-style sweeps).
- Origin: the papers.
- Cost: a per-branch scale constant in `GateBlock` or a norm bound on the modulator; then a run at lr 8e-3 and warmup 120.
- Risk: our zero-init gates already damp the warmup (max grad norm 18 → 8.4); the gain is bounded by the 380 steps of reduced lr.

### 10. Mixture of experts in the shared FFN

Replace the shared FFN by a few experts with a router; the loop then applies different experts at different depths. The gains the looped-MoE papers report are at 6–20B parameters.

- Sources: MoEUT, [arXiv 2405.16039](https://arxiv.org/abs/2405.16039) §8.7: at 244M a naive Universal Transformer is 8× slower per batch than the unshared model and MoEUT 1.7×; Loopie, [arXiv 2607.16051](https://arxiv.org/abs/2607.16051); Sparse Layers, [arXiv 2605.09165](https://arxiv.org/abs/2605.09165).
- Origin: the papers.
- Cost: large (router, load-balancing loss, grouped matmuls); not for a 45-minute budget.

## C. Demoted or recipe-changing

### 11. Adaptive depth per token

Route each token out of the loop when it no longer needs more iterations, compute attention only among the active tokens. Promising at first sight; the literature at our scale is against it.

- Sources: Mixture-of-Recursions 2507.10524 §3.2: underperforms the vanilla model at 135M ("recursive capacity bottleneck"), matches or exceeds it above 360M; its 2.06× throughput is inference with depth-wise batching (§3.3). GRT 2608.15062 §4: MoR trails by 0.08–0.16 nats at every scale under isoFLOPs. Mixture-of-Depths, [arXiv 2404.02258](https://arxiv.org/abs/2404.02258): matches the baseline at equal FLOPs and wall-clock, so its payoff is inference; the norm-agnostic variant [arXiv 2606.16112](https://arxiv.org/abs/2606.16112) finds 20–25% skip rates match full depth at equal training compute, the best training-time case. Halting heads do not realize the savings in batches ([arXiv 2610.32246](https://arxiv.org/abs/2610.32246), [arXiv 2611.02333](https://arxiv.org/abs/2611.02333)). Think-at-Hard, [arXiv 2511.08577](https://arxiv.org/abs/2511.08577), is a LoRA fine-tune of Qwen3 with a decider, not pretraining.
- Only a capacity-routed design (top-k tokens per loop, static shapes) fits our compiled graphs, and at 60–85M the predicted loss penalty must be outweighed by the saved compute.

### 12. Truncated backward through the last B of L loops

Run all L loops forward, detach the state before the last B, backpropagate through B only. Per-token training cost ∝ L + 2B, so (4, 2) costs 8 units against 12 for (4, 4), ~30% of the FLOPs, and the stored activations fall with B.

- Sources: Geiping 2502.05171 §3.3 (k = 8 of mean 32); Parcae, [arXiv 2610.38422](https://arxiv.org/abs/2610.38422) §4, §5.4 (144M, close to our token counts): at matched training time a 4-loop backward beats 8 by 0.011–0.020 nats across five seeds; at a 5× larger budget the preference flips. 2610.26030 §5.1: with the forward loop count fixed, B* = 2 at L = 8 and a predicted 34% compute saving; §5.2: with both free, full backprop wins. 2604.21106 §5.1: truncation lowers the worth of a loop from r^0.46 to r^0.38, and the (L = 2, B = 1) model is the documented failure case.
- Useless for 6×2; for a 3×4 or 2×6 shape it is the one real FLOPs lever. Changes the recipe.

### 13. Batch loop positions together

Shared weights let loop i of one sample set and loop j of another run in one kernel; in our step the long route's first loop and the short route could run as one batch of 2B with per-sample conditioning.

- Source of the mechanism: Relaxed Recursive Transformers, [arXiv 2410.20672](https://arxiv.org/abs/2410.20672) §2.4 (inference, 2–3× with early exit). The training-time use is ours.
- Estimate: under 10%: at 32K tokens per matmul the kernels already fill the GPU, and the B200 measurement says the limit is memory-bound work.

### 14. Local shortcut consistency instead of a second route

LoopFormer took its (t, Δt) conditioning from Shortcut Models, [arXiv 2410.12557](https://arxiv.org/abs/2410.12557), where the self-consistency target is formed locally: one step of size 2d from an intermediate state is matched to two steps of size d from the same state, with no second trajectory. The analogue: from the long route's state at a random loop m, run one block application with Δt = 2/L and match stopgrad(h_{m+2}); +1 loop application per step instead of +L/2, and one lm_head/loss pass instead of two. LoopFormer's own accounting for the dual trajectory is 1.5× FLOPs and 1.3× wall-clock (2602.11451 §9.1).

- Origin: the literature agent's extrapolation of the Shortcut Models objective.
- It drops the short route's cross-entropy and replaces the short route, both on the ruled-out list; recorded for completeness.

## Context from the scaling literature

- 2604.21106 §4.3: looping a block r times is worth r^0.46 unique blocks at matched compute, so a 410M model looped 4× matches a 580M plain model while paying the training cost of a 1B one; truncated backprop lowers the exponent to 0.38, hyper-connections raise it to 0.65.
- 2610.26030 §6.2–6.3: with the model size free, "the law favors no looping"; the looped model breaks even at (2, 2) with 27–32% fewer parameters. Its empirical coefficient at (2, 2), 239M, 4B tokens is 1.87 (a block visited twice is worth 1.87 blocks) against 2604.21106's 1.38, so the worth of a loop depends strongly on the sandwich layout and the token budget.
- Loopie 2607.16051 §2.4, §2.8: the marginal return of recurrence decays rapidly at matched compute; its own gain is "entirely attributable to the doubled microbatch size" that the saved memory allows (our 4× batch → 1.25× tokens/s is the counter-evidence here). SMELT §13.5: 3 and 4 loops regress against 2 because matched FLOPs force a thinner model. Meta 2610.40325: gains "eventually saturate, reverse, or become unstable".
- LoopFormer 2602.11451 §9.1, App B.3: the dual-trajectory objective is ~1.5× FLOPs and ~1.3× wall-clock (ours: 1.26–1.4×); the elastic training itself costs ~0.03 nats at matched compute against a fixed-loop model. §4.4: at a fixed budget M the schedule choice spreads perplexity by 1.4–3 points, best schedules take larger steps early.
- Where looping does win on loss: data-limited pretraining ([arXiv 2611.06206](https://arxiv.org/abs/2611.06206), [arXiv 2608.26973](https://arxiv.org/abs/2608.26973), 2610.33360: the cost of weight tying is smallest at low token budgets) and reasoning benchmarks rather than perplexity ([arXiv 2502.17416](https://arxiv.org/abs/2502.17416); Hyperloop §4.2). OpenWebText at 1.2B tokens is neither.

## Does not help (with the source that says so)

- Truncated backprop at L = 2 (2604.21106 §5.1: the r = 2, r_bwd = 1 outlier).
- Per-loop LoRA: our ablation (no gain, +9% step); RRT ties only at small scale (GRT §4); Meta 2610.40325: adapter gains appear only "as more loops are added".
- A random initial state instead of the embedding (Geiping's recipe; no gain reported for a 2-loop model).
- Hyper-connections at every layer (mHC: −35% throughput; once per loop is enough).
- More than two loops for a dense model at matched compute (SMELT §13.5, 2610.26030, LPT: vanilla looping stops improving beyond 3).
- Additive-only conditioning: our ablation, +0.023 nats.
- The consistency loss alone, without the short route's cross-entropy (LoopFormer App B).
- The plain model's hyperparameters unchanged: Ouro §4.3 and our warmup finding.
- Expecting knowledge or perplexity gains from loops; the wins are on reasoning tasks (Hyperloop §4.2, 2502.17416).
- Halting heads or routers that stop per token (2610.32246: tokens run almost every iteration in batches anyway).
- Loopie's recipe as a throughput lever: it is the bigger batch (ruled out).
- Cut Cross-Entropy, [arXiv 2411.09009](https://arxiv.org/abs/2411.09009): logits never materialized, loss memory to ~1 MB; same class as the fused attention kernel (helps the plain model equally), though it would help our two-route step twice.
- From first principles, considered and dropped: a prefix shared between the two routes (the routes diverge at the first step size), chunked cross-entropy (memory, not time), weight residency in cache (14 MB weights vs 50 MB activations), a sliding window or half context in later loops (changes the model more than it saves), a wider d_ff for the shared block (helps the plain model too), a larger vocabulary or FP8 (same), a smaller batch (fewer tokens), exact reuse of the first loop's output under a gate (the gate would learn to skip the loop), and pooling the sequence in later loops (changes what a loop is).

## Suggested order

For one more B200 run of 6×2 at batch 256, the cheapest changes with the most consistent evidence are 6 (prelude–loop–coda) and 8 (per-token gate with the hidden state as input), then 1 (shared first-loop K/V) as the one change that both saves work and improved loss when trained from scratch, then 7 (hyper-connections once per loop). Ideas 2–5 are the only real tokens-per-second levers and are untested in the literature; 2 and 3 should be measured on a 3×4 shape first, where the saving is largest. Expect the sum to close part of the 0.15-nat gap; no compute-matched study has a dense two-loop model ahead of the plain one.
