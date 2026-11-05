# LoopFormer conditioning ablations — implementation plan

Design: [2026-11-04-loopformer-conditioning-ablations.md](../designs/2026-11-04-loopformer-conditioning-ablations.md)

## Goal

Add the `gate` and `additive` conditioning variants and per-loop LoRA to the LoopFormer, run each as a 15-minute A100 run of 3×4, and record the comparison with the phase-1 baseline.

## Decomposition

- `model.py` — `ModelConfig` gains `conditioning` ("adaln" default, "gate", "additive") and `loop_lora_rank` (0 default). The block builds its norms, modulator and LoRA factors from them; `LoopFormerLM.forward` adds c to the hidden state at each loop start for `additive`, and passes the loop position to the blocks for LoRA. `muon_parameters` leaves the LoRA factors to AdamW.
- `run.py` — `RunConfig` gains the two fields with defaults and passes them into `ModelConfig`.
- `train.py` — the two flags.
- `configs/` — nothing new: the flags go through the extra arguments of `a100-20min.sh`.
- `tests/` — the variant tests of the design.
- `README.md` — an "Ablations" results section.

## Behaviours to verify

1. Given any variant and any M, when one token changes, then the logits before it do not change.
2. Given any variant, when a trained model is saved and loaded into a fresh model, then the logits are identical for every M (the LoRA factors included).
3. Given a fresh `gate` or `additive` model, when it runs along any schedule, then its logits equal those of the same weights with no conditioning at all: the block passes its input through (`gate`) or c is zero (`additive`).
4. Given any variant, when training runs 200 steps on one sample, then the loss at the full route falls below 0.05.
5. Given a LoRA model, when its parameters are split for the optimizers, then no LoRA factor is in Muon's list.
6. Given the three runs, when they end, then `README.md` compares them with the phase-1 3×4 run on final loss, per-loop curve, peak memory, ms per step and max gradient norm.

## Milestones

1. The variants in the model, with flags and tests.
   Done when: `uv run pytest` inside `loopformer/` is green with the new tests, and the three variants run for a minute each on the Mac.
   Verified by: behaviours 1–5.
2. The three A100 runs and the README section.
   Done when: `budget_eval.json` of all three is fetched and the "Ablations" section is in `README.md`.
   Verified by: behaviour 6.

## Assumptions

- The phase-1 3×4 run's results are in `loopformer/runs/modal/owt-a100-loop-3x4-20min/` and on the volume: loss 3.879, 46.0 GB, 417 ms per step, max gradient norm 18.1.
- `configs/a100-20min.sh` passes extra arguments last, so `--conditioning gate` overrides nothing and adds the variant.
- `muon_parameters` selects by `p.ndim == 2` inside the blocks; the LoRA factors are 2-D, so the rule must exclude them by name or attribute.
- The B200 phase-2 run is in flight on the committed code; the model changes keep the defaults, so its evaluation on the volume still loads.

## Open questions

- None.
