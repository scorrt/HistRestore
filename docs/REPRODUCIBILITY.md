# Reproducibility Notes

## Candidate-Bank Protocols

Two MixedDoc protocols are retained:

1. A lightweight candidate bank without official MMDIR predictions, used for ablation and diagnostic analyses.
2. An MMDIR-augmented candidate bank with `mmdir_official`, used for the main MixedDoc comparison.

See:

- `results/mixedoc/protocol_delta_audit.md`
- `results/mixedoc/main_sota_table_mixeddoc.csv`

## Released Result Audit

The public package is designed to support aggregate-result verification without redistributing raw benchmark images. The fastest audit entrypoint is:

```bash
python scripts/summarize_released_results.py
```

End-to-end regeneration of every manuscript table requires:

- raw datasets obtained from the original providers;
- third-party restoration model checkpoints or official predictions;
- precomputed candidate images or candidate metrics;
- Qwen3-VL review logs for experiments involving semantic priors.

## Hardware Environment

The reported experiments used NVIDIA A100 80GB GPUs for the GPU runtime measurements and Qwen3-VL review service. Runtime claims in the released tables should be interpreted under that hardware setting.

## Main Result Files

- Historical-537 main grouped evaluation: `results/historical537/historical537_group_main_results.csv`
- Historical-537 semantic-prior comparisons: `results/historical537/semantic_compatibility_heldout_summary.csv`
- MixedDoc Protocol-B main table: `results/mixedoc/main_sota_table_mixeddoc.csv`
- MixedDoc paired bootstrap: `results/mixedoc/mmdir_augmented_bootstrap.csv`
- Runtime summary: `results/runtime_summary.csv`
