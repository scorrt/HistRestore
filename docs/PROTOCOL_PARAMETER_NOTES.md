# Protocol and Parameter Notes

## Compound-proxy

The synthetic subset contains 120 samples from 60 clean Jung source pages, with two degraded variants per source page. Generation uses seed `20260805`, maximum long side `1600`, one to three sampled operators with probabilities `0.30/0.50/0.20`, severity sampled uniformly from `0.25` to `0.90`, and JPEG compression with probability `0.35` at quality `35–75`.

## MixedDoc utility

The MMDIR-augmented protocol uses `classical_shadow_0.40` as the baseline and the fixed utility coefficients:

```text
lambda_SSIM = 8.0
lambda_R = 0.55
lambda_H = 0.15
```

The corresponding eight-candidate bank is stored in `configs/mixeddoc_candidate_banks.json`.
