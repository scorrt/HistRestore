# Reproducibility Notes

## Historical-537

The final split contains 424 training pages and 113 source-group-held-out pages. Variants derived from the same source page remain in the same partition. The exact manifest is `splits/historical537_group_split_manifest.csv`.

The nested candidate order for K = 6, 10, 14, 18, and 22 is `configs/historical537_nested_candidate_order.csv`. Per-page candidate measurements and final evidence-only/direct-review selections are under `results/per_page/`.

## MixedDoc

The fixed split contains 1083 training, 377 validation, and 377 test pages. `configs/mixeddoc_candidate_banks.json` defines the seven-candidate lightweight bank and the eight-candidate MMDIR-augmented bank. The main test results are `results/mixedoc/main_results.csv` and `results/mixedoc/paired_bootstrap.csv`.

## VCCRP

`src/histrestore/evidence.py` implements the manuscript definition:

```text
0.45 * (1 - edge_jaccard)
+ 0.25 * min(4 * foreground_shift, 1)
+ 0.20 * min(3 * contrast_shift, 1)
+ 0.10 * min(4 * mean_shift, 1)
```

## Region-aware candidate

`src/histrestore/region.py` implements the fixed spatial operating point. The raw alpha map is clipped to `[0.68, 0.98]`, Gaussian-smoothed with `sigma = 3`, and then applied to the source-to-DocRes residual.

## Hardware

GPU runtime measurements use NVIDIA A100 GPUs with 80 GB memory per GPU. Direct Qwen3-VL review uses two-GPU tensor parallelism.
