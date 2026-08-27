# Reproducibility Notes

## Candidate-Bank Protocols

Two MixedDoc protocols appear in the experiments:

1. Lightweight candidate bank without MMDIR official predictions.
2. MMDIR-augmented candidate bank with `mmdir_official`.

The manuscript should use the MMDIR-augmented protocol as the main MixedDoc comparison. The lightweight protocol is retained as an ablation because it isolates evidence-aware selection when only classical, DocRes, blend, and region candidates are available.

See:

- `results/mixedoc/protocol_delta_audit.md`
- `results/mixedoc/main_sota_table_mixeddoc.csv`

## Main Scripts

- `scripts/evaluate_mmdir_official_mixeddoc.py`: evaluates official MMDIR predictions under the same PSNR/SSIM/VCCRP protocol.
- `scripts/train_mmdir_augmented_selector.py`: trains/evaluates the MMDIR-augmented selector.
- `scripts/build_mmdir_frozen_compat_balanced_20260827.py`: evaluates raw prior versus frozen compatibility-balanced prior in Protocol B.
- `scripts/build_frozen_compat_external_20260827.py`: validates frozen compatibility features on MixedDoc and OSR.
- `scripts/build_semantic_compatibility_variants_20260827.py`: Historical-537 semantic compatibility variant analysis.

## Important Limitation

Some scripts require precomputed candidate metrics, official MMDIR predictions, or Qwen review logs. These are not fully redistributed because they can contain third-party data or generated image-derived artifacts. The final CSV summaries are provided for aggregate verification.

