# Supplementary Appendix S1: Prompt, Evidence, and Reproducibility Details

This appendix documents the supplementary materials referenced by the manuscript
"HistRestore: Evidence-Constrained Restoration Selection with Vision-Language
Priors for Historical Document Images".

## S1. Structured Qwen Review Interface

The structured review module uses Qwen3-VL-32B-Instruct-FP8 only at the
selection stage. The model does not generate or edit image pixels. Its output is
restricted to categorical labels and a short rationale that are converted into
features for candidate selection.

The review prompt supplies:

- the source page or candidate panel;
- page-level visual statistics and source-candidate evidence;
- content-preservation criteria;
- layout-compatibility criteria;
- a fixed output schema.

The expected JSON fields are:

```json
{
  "decision": "use_docres | blend_docres | use_classical | preserve_input | manual_review",
  "degradation": "shadow | stain | bleed | blur | binarization | complex",
  "policy": "preserve | deshadow | binarize | blend",
  "strength": "none | light | medium | strong",
  "content_risk": "low | medium | high",
  "reason": "short evidence-based explanation"
}
```

Only the structured fields are used by the selection model. Free-form rationale
text is retained for audit but is not used as a direct numerical feature.

## S2. Label Mapping for Prior Distillation

For train-only prior distillation, Qwen decisions on reviewed training pages are
mapped to candidate families:

| Qwen decision | Candidate family used for supervision |
|---|---|
| `use_docres` | `docres_deshadow` |
| `blend_docres` | `docres_input_blend_0.90` |
| `use_classical` | `classical_shadow_0.40` |
| `preserve_input` | `input` |
| `manual_review` | `classical_shadow_0.40` |

The distilled prior is trained as a GradientBoostingClassifier over observable
panel/evidence features on training pages. At deployment, it predicts a
candidate-family probability vector. No held-out test-page Qwen JSON is used in
the train-only prior-distillation protocols.

## S3. Evidence Features

The source-candidate evidence vector is computed for each candidate by comparing
the candidate image with the degraded source page. The implementation is in:

- `src/histrestore/evidence.py`
- `src/histrestore/selector.py`
- `src/histrestore/semantic_prior.py`

The principal evidence fields are:

| Field | Meaning |
|---|---|
| `edge_jaccard` | Jaccard overlap between source and candidate edge masks |
| `edge_keep` | Fraction of source edges retained by the candidate |
| `foreground_shift` | Absolute foreground-area change |
| `mean_shift` | Absolute normalized mean-intensity change |
| `contrast_shift` | Absolute normalized contrast change |
| `contrast_after` | Candidate grayscale standard deviation |
| `sharp_after` | Candidate Laplacian-variance sharpness |
| `content_risk` | Visual Content-Change Risk Proxy (VCCRP) |

The VCCRP implementation uses the same source-relative visual-change terms as
the manuscript:

```text
VCCRP =
  0.45 * rho(edge_discontinuity)
  + 0.25 * rho(foreground_shift)
  + 0.20 * rho(contrast_shift)
  + 0.10 * rho(mean_shift)
```

where `edge_discontinuity = 1 - edge_jaccard`, and `rho` is a saturating
normalizer. Lower VCCRP indicates less source-relative visual change.

## S4. Candidate-Aware Compatibility Features

The candidate-aware semantic compatibility features convert a page-level prior
into candidate-specific evidence before utility estimation. They depend only on
candidate family, predicted prior probabilities, candidate strength, and
content-risk evidence.

The public implementation reports features such as:

- `prior_candidate`;
- `prior_family`;
- `top_candidate_match`;
- `top_family_match`;
- `strength_gap`;
- `prior_confidence`;
- `prior_entropy`;
- `risk_candidate_strength`;
- `mild_prior_under_risk`;
- `strong_prior_risk_conflict`;
- `candidate_prior_after_risk`;
- `family_prior_after_risk`.

## S5. Supplementary Tables and Files

The supplementary package includes:

- `results/historical537/statistical_tests_report.md` for paired
  Historical-537 statistics;
- `splits/historical537_group_split_manifest.csv` and
  `splits/historical537_group_split_counts.csv` for Table S3;
- `configs/historical537_22_candidate_names.csv` for the 22-candidate order;
- `results/historical537/semantic_compatibility_heldout_summary.csv`;
- `results/historical537/semantic_compatibility_paired_stats.csv`;
- `results/mixedoc/frozen_compat_balanced_summary.csv`;
- `results/mixedoc/frozen_compat_balanced_paired_stats.csv`;
- `results/external/frozen_compat_external_summary.csv`;
- `results/external/frozen_compat_external_paired_stats.csv`;
- `results/runtime_summary.csv`;
- protocol notes in `docs/PROTOCOL_PARAMETER_NOTES.md`;
- reproducibility notes in `docs/REPRODUCIBILITY.md`.

Raw datasets, third-party model weights, and third-party prediction archives are
not redistributed. They should be obtained from the original dataset and model
providers cited in the manuscript.
