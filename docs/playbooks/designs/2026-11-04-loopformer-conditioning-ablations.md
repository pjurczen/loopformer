# LoopFormer conditioning ablations

Design: a second slice of [the LoopFormer experiment](2026-11-03-loopformer.md).

## Context

Phase 1 showed that the LoopFormer block costs more activation memory than a plain block: 46 GB against 37 GB at batch 64, which kept the B200 run from the plain model's batch 256. The extra memory comes from the adaLN modulation: the (1 + γ) scale after each norm and the gate α on each residual branch, two saved tensors per branch.

A literature review found lighter ways to tell the shared block where it is in the loop. TMLT's theory and the adaLN-Zero analysis say the parts that matter are a multiplicative per-loop signal and the zero start. Universal Transformers add the step as an embedding. Relaxed Recursive Transformers give each loop its own low-rank weight deltas, which also attacks the quality gap to the plain model. This slice measures three of these against the phase-1 3×4 run.

## Decisions

- One change per run, all on the 3×4 shape with the phase-1 flags, 15 minutes on an A100-80GB. The phase-1 3×4 run is the baseline: loss 3.879, per-loop 4.110 / 3.928 / 3.893 / 3.879, 46.0 GB, 417 ms per step.
- The model gets a `conditioning` setting with three values and an optional per-loop LoRA rank. The default values reproduce the current model, so older runs load unchanged.
- The LoRA factors go to AdamW, not Muon. Muon's orthogonalized steps have a fixed size, which would move the zero-start B factors too hard.

## The variants

- `adaln` (default): the current block. Weight-free norms, modulator → (γ₁, γ₂, α₁, α₂).
- `gate`: the plain pre-norm block (norms with their learned weight) plus a zero-start gate α per branch. The modulator outputs 2d.
- `additive`: the plain pre-norm block, no modulator. The vector c is added to the hidden state once at the start of each loop, before the k blocks. The last layer of each conditioning MLP starts at zero, so c = 0 at step 0 and a fresh model equals a plain looped model.
- `loop_lora_rank` r (0 = off): each shared matrix (qkv, output, w1, w2, w3) gets one low-rank pair per loop position i = 1 … L, Aᵢ of r × d_in and Bᵢ of d_out × r. Loop i adds Bᵢ Aᵢ x. B starts at zero. Rank 32 adds about 0.4M weights per block per loop, 5M for 3×4.

Everything else stays: the trajectories, the two-route loss, the evaluation, the decoding. `RunConfig` gains `conditioning` and `loop_lora_rank` with defaults.

## Flags

`--conditioning adaln|gate|additive` and `--loop_lora_rank N` on `train.py`.

## Tests

On the tiny model, for each variant:

- No leakage from the future, at every M.
- The checkpoint round-trip gives identical logits; the LoRA factors must survive it.
- A fresh `gate` model returns its input unchanged, so every route gives the same logits. A fresh `additive` model ignores the timing of its loops: two schedules with the same loop count give the same logits.
- `Linear` gains an optional per-loop LoRA, the one change to a copied component in this slice.
- Training on one sample drives the loss below 0.05 in 200 steps.

## The runs

| Run | Conditioning | Per-loop weights | Question |
|---|---|---|---|
| `owt-a100-loop-3x4-20min` (done) | adaln | shared | the baseline |
| `owt-a100-loop-3x4-gate-20min` | gate | shared | does dropping the scales cost anything? |
| `owt-a100-loop-3x4-additive-20min` | additive | shared | does dropping the modulation cost anything? |
| `owt-a100-loop-3x4-additive-lora32-20min` | additive | + LoRA rank 32 | do a few per-loop weights close the gap? |

The runs compare on the final loss, the per-loop curve, the peak memory, the ms per step and the max gradient norm, recorded in the README under "Ablations".

## Out of scope

- Truncated backprop through loops, sandwich norms and DeepLoop residual scaling.
- A LoRA variant on top of `gate`. It is one more run if `additive` is clearly worse than `gate`.
