v0.1.0 - 2025-08-24
=====================

Summary
-------
- Added optional per-gene z-scoring for MLM experiments.
- Restored backward compatibility for `SCRNALongformer.forward` when the optional MLM head is not enabled (returns `(logits, emb)` by default).
- Updated training script to support both classifier and MLM modes and added a smoke test that prepares its own data artifact for CI.

Tag: `v0.1.0`

Notes
-----
- No new runtime dependencies were added.
- The repository now includes an example experiment config `configs/exp_zscore.yaml`.
- Large binary artifacts (e.g. `.pt`) are not tracked in the repository going forward; consider cleaning history if desired.
