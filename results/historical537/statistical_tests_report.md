# Historical-537 Paired Tests and Held-Out Split Audit

## Paired Statistical Tests

Comparison: Distilled-prior HistRestore vs the strongest fixed blend (`docres_classical040_blend_0.85`) on the same 113 source-page held-out pages.

| Metric | Mean diff | Bootstrap 95% CI | paired t p | Wilcoxon p | Cohen dz | Rank-biserial | Win rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| PSNR | 0.430353 | [-0.0040, 0.8994] | 0.0642085 | 0.048608 | 0.1758 | 0.2283 | 0.4690 |
| SSIM | 0.005984 | [-0.003074, 0.016532] | 0.236797 | 0.78812 | 0.1119 | 0.0311 | 0.4513 |
| VCCRP | 0.006030 | [-0.010961, 0.025443] | 0.515457 | 0.33016 | 0.0614 | 0.1127 | 0.3894 |

## Held-Out Source Distribution

| Dataset source | Total | Train | Held-out | Held-out / source | Held-out / 113 |
|---|---:|---:|---:|---:|---:|
| compound_proxy | 120 | 90 | 30 | 0.2500 | 0.2655 |
| dibco18 | 10 | 7 | 3 | 0.3000 | 0.0265 |
| dibco19 | 20 | 19 | 1 | 0.0500 | 0.0088 |
| jung | 87 | 72 | 15 | 0.1724 | 0.1327 |
| kligler | 300 | 236 | 64 | 0.2133 | 0.5664 |
| TOTAL | 537 | 424 | 113 | 0.2104 | 1.0000 |

Kligler accounts for 64/113 held-out pages (56.6%), close to its full-set share of 300/537 (55.9%). This does not indicate an inflated Kligler-only test set.

## Files

- `historical537_full_vs_best_fixed_blend_paired_rows.csv`
- `historical537_full_vs_best_fixed_blend_stat_tests.csv`
- `historical537_split_source_distribution.csv`
