## v0.1.0 - 2025-08-24

### Added
- Optional per-gene z-scoring (`data.zscore`) and `configs/exp_zscore.yaml` to enable it.
- Masked-gene (MLM) regression support with an optional `mlm` flag in the model and training script.

### Fixed
- Backwards compatibility: `SCRNALongformer.forward` returns `(logits, emb)` when MLM not enabled. Training/eval adjusted to support both return shapes.
- Fixed CI flaky test by ensuring the mlm smoke test prepares its own `.npz` artifact.

### Testing
- `pytest` suite updated and passing locally (10 tests including MLM smoke + zscore test).

### Notes
- Large binary artifacts should not be committed; they are ignored now via `.gitignore` and a tag `v0.1.0` was created for this release.
