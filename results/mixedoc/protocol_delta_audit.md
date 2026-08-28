# MixedDoc Protocol Difference Audit

## Conclusion

The lower 20.x lightweight-bank results and the 27.x MMDIR-augmented results are not contradictory. They use the same 377-page held-out test split, but different candidate banks.

- The lightweight protocol uses classical, DocRes, blend, and region-aware candidates.
- The MMDIR-augmented protocol additionally includes official MMDIR predictions as a strong candidate.
- The released main MixedDoc comparison uses the MMDIR-augmented protocol.

The score difference is therefore caused by candidate-bank coverage rather than by a split change or metric mismatch.

## Verified Same Split

Both protocols use the stable-hash MixedDoc split released in:

- `splits/mixeddoc_stable_hash_split_manifest.csv`
- `splits/mixeddoc_stable_hash_split_counts.csv`

The test split contains 377 pages.

## Lightweight Candidate Bank

Split:

- train: 1083
- val: 377
- test: 377

Candidate bank:

- `input`
- `classical_shadow_0.40`
- `docres_deshadow`
- `docres_input_blend_0.90`
- `docres_input_blend_0.80`
- `docres_classical040_blend_0.90`
- `region_controller`

Test results:

| Method | PSNR | SSIM | Risk |
|---|---:|---:|---:|
| Evidence-only selector | 20.3113 | 0.73146 | 0.20462 |
| Distilled-prior selector | 20.3419 | 0.73195 | 0.20504 |
| Estimated oracle of this bank | 20.8415 | - | - |

This protocol is retained as an ablation setting because it isolates evidence-aware selection when only lightweight restoration candidates are available.

## MMDIR-Augmented Candidate Bank

Released result files:

- `results/mixedoc/main_sota_table_mixeddoc.csv`
- `results/mixedoc/mmdir_augmented_bootstrap.csv`
- `results/mixedoc/mmdir_augmented_split_summary.csv`

Split:

- train: 1083
- val: 377
- test: 377

Candidate bank:

- `input_degraded`
- `classical_shadow_0.40`
- `docres_deshadow`
- `docres_input_blend_0.90`
- `docres_input_blend_0.80`
- `docres_classical040_blend_0.90`
- `region_controller`
- `mmdir_official`

Test results:

| Method | PSNR | SSIM | Risk |
|---|---:|---:|---:|
| Input degraded | 17.0274 | 0.70449 | 0.00000 |
| DocRes deshadow | 20.2725 | 0.73145 | 0.21840 |
| MMDIR official | 27.5196 | 0.94791 | 0.43635 |
| HistRestore + MMDIR | 27.6673 | 0.94797 | 0.43058 |
| Oracle candidate pool | 27.8588 | 0.94593 | 0.42305 |

Paired bootstrap for HistRestore + MMDIR versus MMDIR official:

| Metric | Mean delta | 95% CI |
|---|---:|---:|
| PSNR | +0.1477 dB | [+0.0381, +0.2681] |
| SSIM | +0.000059 | [-0.000770, +0.000797] |
| Risk | -0.00578 | [-0.00862, -0.00323] |

This is the released main MixedDoc protocol because it compares against the current strong MixedDoc restoration output under the same split and evaluation code.

## Why the Score Changes

The added MMDIR official prediction is the main reason for the numerical shift. On the full MixedDoc evaluation pool, MMDIR official predictions are much stronger than the DocRes-only candidates. The two score ranges therefore correspond to different candidate-bank protocols, not different conclusions.
