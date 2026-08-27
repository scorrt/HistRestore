# MixedDoc 20.x vs 27.x Protocol Difference Audit

## Conclusion

The 20.x and 27.x MixedDoc results are not contradictory. They use the same 377-page held-out test split, but different candidate-bank protocols.

- The 20.x results are from the DocRes/classical/region lightweight candidate bank.
- The 27.x results are from the MMDIR-augmented candidate bank, where official MMDIR predictions are added as a strong candidate.
- The 377 test sample IDs are identical across the two protocols.

Therefore, the score jump is caused by candidate-bank coverage, mainly the inclusion of MMDIR official outputs, not by a split change or metric mismatch.

## Verified Same Split

Compared files:

- `outputs/mixeddoc_qwen_prior_split_sweep_20260812/sw24_rp1p2_hp0p35/test_qwen_prior.jsonl`
- `outputs/mixeddoc_qwen_prior_split_sweep_20260812/sw24_rp1p2_hp0p35/test_no_qwen.jsonl`
- `outputs/mmdir_augmented_selector_20260813/rp0p55_hp0p15_sw8/test_histrestore_mmdir_prior_selector.jsonl`
- `outputs/mmdir_augmented_selector_20260813/rp0p55_hp0p15_sw8/test_mmdir_official.jsonl`

All four files contain exactly 377 unique sample IDs. The ID sets are identical.

## Protocol A: DocRes-Only Lightweight Candidate Bank

Source:

- `outputs/mixeddoc_qwen_prior_split_sweep_20260812/sw24_rp1p2_hp0p35/summary.json`
- Candidate metrics: `outputs/mixeddoc_candidate_cache_plus_region_20260812/candidate_metrics.jsonl`

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
| No-Qwen selector | 20.3113 | 0.73146 | 0.20462 |
| Qwen-prior selector | 20.3419 | 0.73195 | 0.20504 |
| Estimated oracle of this bank | 20.8415 | - | - |

Interpretation:

This protocol evaluates whether HistRestore can make adaptive source-relative decisions when only lightweight restoration candidates are available. It is useful for ablation, risk-aware selection, region-controller discussion, and Qwen-prior diagnostics. It should not be used as the main MixedDoc SOTA table after MMDIR outputs are available.

## Protocol B: MMDIR-Augmented Candidate Bank

Sources:

- `outputs/mmdir_same_protocol_all_20260813/summary.json`
- `outputs/mmdir_augmented_selector_20260813/rp0p55_hp0p15_sw8/summary.json`
- `outputs/paper_readiness_20260813/main_sota_table_mixeddoc.csv`

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
| HistRestore + MMDIR prior | 27.6673 | 0.94797 | 0.43058 |
| Oracle candidate pool | 27.8588 | 0.94593 | 0.42305 |

Paired bootstrap for HistRestore + MMDIR prior vs MMDIR official:

| Metric | Mean delta | 95% CI |
|---|---:|---:|
| PSNR | +0.1477 dB | [+0.0406, +0.2691] |
| SSIM | +0.000059 | [-0.000770, +0.000797] |
| Risk | -0.00578 | [-0.00862, -0.00323] |

Interpretation:

This is the paper-ready MixedDoc main protocol because it compares against the current strong MixedDoc restoration output under the same split and evaluation code. The claim should be: HistRestore improves archival acceptance over a strong professional candidate by selecting safer alternatives for a small subset of pages.

## Why the Score Changes from 20.x to 27.x

The key reason is the added MMDIR official prediction:

- Full 1837-page same-protocol MMDIR official average: 27.5257 dB / 0.94753 SSIM.
- Full 1837-page DocRes deshadow average: 20.5167 dB / 0.73703 SSIM.

This alone explains the large numerical shift. The two result ranges are different candidate-bank protocols, not different conclusions.

## Paper Decision

Use the MMDIR-augmented table as the main MixedDoc comparison table.

Keep the DocRes-only 20.x protocol as supplementary or ablation evidence, with a clear label such as:

> Lightweight candidate-bank protocol without MMDIR official predictions.

Do not mix 20.x and 27.x numbers in one main table unless the candidate bank column is explicit.

Recommended main-text wording:

> For the main MixedDoc comparison, we use the MMDIR-augmented candidate bank because official MMDIR predictions are available for the benchmark. The earlier DocRes-only bank is retained as a lightweight-bank ablation to isolate the effect of evidence-aware selection and VLM prior without a strong MMDIR candidate.

