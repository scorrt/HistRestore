# Supplementary File Index

This index maps the manuscript Supplementary Materials statement to the released repository files.

## Appendix S1

- Qwen3-VL Prompt A–E templates and JSON schemas: `docs/Supplementary_Appendix_S1.md`
- Label mapping used for prior distillation: `configs/qwen_label_mapping.json`

## Tables S1–S3

- **Table S1 — Paired Historical-537 statistics:** `results/historical537/table_s1_paired_statistics.csv`
- **Table S2 — Nested candidate order for K = 6, 10, 14, 18, 22:** `configs/historical537_nested_candidate_order.csv`
- **Table S3 — Historical-537 split manifest:** `splits/historical537_group_split_manifest.csv` and `splits/historical537_group_split_counts.csv`

## Additional supplementary results

- Restricted-bank NR-IQA: `results/nriqa/historical537_nriqa_full22_restricted16.csv`
- Source-wise robustness: `results/historical537/source_wise_robustness.csv`
- Selector-family robustness: `results/selector_family/historical537_selector_family_results.csv` and `results/selector_family/historical537_mlp_10seeds_summary.csv`
- Candidate-bank size sensitivity: `results/sensitivity/candidate_bank_size_sensitivity.csv`
- VCCRP coefficient sensitivity: `results/sensitivity/historical537_vccrp_coefficient_sensitivity.csv`
- Semantic gating: `results/semantic/historical537_semantic_gating_summary.csv`
- Semantic compatibility ablation: `results/semantic/compatibility_ablation_summary.csv` and `results/semantic/compatibility_ablation_paired_stats.csv`
- MixedDoc lightweight/MMDIR-augmented audit: `results/mixedoc/`
- OSR/external semantic-prior analysis: `results/semantic/osr_semantic_prior_summary.csv`
- OCR diagnostics: `results/ocr/`
- Runtime measurements: `results/runtime/runtime_summary.csv`
- Per-page Historical-537 audit data: `results/per_page/`
