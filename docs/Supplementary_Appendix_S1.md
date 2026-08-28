# Supplementary Appendix S1: Qwen3-VL Prompt Templates and JSON Schemas

## S1.1 Inference settings

- Model: `Qwen3-VL-32B-Instruct-FP8`
- Temperature: `0`
- Historical-537 direct review: Prompt A
- MixedDoc training-page candidate review: Prompt B
- Compound-budget analysis: Prompt C
- OSR shadow-strength review: Prompt D
- Hard-case pilot: Prompt E

Dynamic page-level statistics and evidence dictionaries are inserted at the placeholders shown below.

## S1.2 Prompt A — Historical-537 degradation prior

### With evidence

```text
You are a conservative visual-language assistant for historical document image restoration. Inspect only the input page image. Do not transcribe text and do not infer missing characters. No-reference visual statistics are provided to constrain the judgment. Stats: {VISUAL_STATS_JSON}. High near_binary_ratio with text-only layout often indicates a binarization benchmark page; high midtone_ratio and high illumination_cv often indicate shadow/uneven lighting; large foreground ratio or visible illustrations/marginalia increase content-risk and favor preserve/blend. Return strict JSON with these fields: {"degradation":"already_clean|shadow|binarization_needed|appearance_noise|complex", "policy":"preserve|deshadow|binarize|blend", "strength":"none|light|medium|strong", "content_risk":"low|medium|high", "reason":"short visual reason"}. Use 'preserve' or 'light' when the page is already readable or when aggressive restoration may remove pale strokes, seals, notes, or paper texture. Use 'binarize' only for near-binary benchmark-like pages where a clean black/white foreground-background target is visually appropriate. Use 'blend' for mixed degradation where no single operation is clearly safe.
```

### Without evidence

```text
You are a conservative visual-language assistant for historical document image restoration. Inspect only the input page image. Do not transcribe text and do not infer missing characters. Return strict JSON with these fields: {"degradation":"already_clean|shadow|binarization_needed|appearance_noise|complex", "policy":"preserve|deshadow|binarize|blend", "strength":"none|light|medium|strong", "content_risk":"low|medium|high", "reason":"short visual reason"}. Use 'preserve' or 'light' when the page is already readable or when aggressive restoration may remove pale strokes, seals, notes, or paper texture. Use 'binarize' only for near-binary benchmark-like pages where a clean black/white foreground-background target is visually appropriate. Use 'blend' for mixed degradation where no single operation is clearly safe.
```

Expected JSON:

```json
{"degradation":"already_clean|shadow|binarization_needed|appearance_noise|complex", "policy":"preserve|deshadow|binarize|blend", "strength":"none|light|medium|strong", "content_risk":"low|medium|high", "reason":"short visual reason"}
```

## S1.3 Prompt B — MixedDoc candidate risk review

```text
You are a conservative visual-language reviewer for historical document restoration. The image panel shows LEFT=input, MID=classical light correction, RIGHT=DocRes candidate. Use the panel and the structured no-reference evidence. Do not transcribe text and do not infer missing characters. Decide whether DocRes should replace the classical candidate for a digital-archive restoration pipeline. Return strict JSON only: {"decision":"use_docres|use_classical|blend_docres|preserve_input|manual_review", "risk":"low|medium|high", "reason":"short visual reason"}. Choose use_docres when the RIGHT image visibly removes shadow, watermark, seal tint, or uneven background while keeping glyph strokes and layout stable. Choose blend_docres when RIGHT is mostly better but slightly too strong. Choose use_classical/preserve/manual_review when RIGHT removes pale strokes, seals, marginalia, texture, or distorts glyph shapes. Structured evidence={STRUCTURED_EVIDENCE_JSON}
```

Expected JSON:

```json
{"decision":"use_docres|use_classical|blend_docres|preserve_input|manual_review", "risk":"low|medium|high", "reason":"short visual reason"}
```

Training-page decisions are mapped using `configs/qwen_label_mapping.json` and used to train the MixedDoc family-prior classifier.

## S1.4 Prompt C — Compound-budget review

```text
You are a reviewer in a historical document image restoration pipeline. The image contains two panels. LEFT is the degraded input; RIGHT is an automatic mild deshadow candidate. Use only visible evidence and the structured metrics. Do not transcribe or hallucinate missing text. Choose one decision from: accept_candidate, keep_input, rollback_candidate, manual_review. The candidate is produced by a conservative restoration model, so accept it when shadows/noise are reduced and strokes, character shapes, reading order, and layout remain visually consistent. Use rollback_candidate only when there is clear visible evidence of stroke loss, character-shape change, text-line deformation, or layout-element deletion. Use manual_review when risk is uncertain rather than automatically rejecting a useful candidate. Return strict JSON with keys decision, confidence, risk, reason. Structured evidence: {COMPACT_EVIDENCE_JSON}
```

Expected JSON:

```json
{"decision":"accept_candidate|keep_input|rollback_candidate|manual_review", "confidence":"number or short value", "risk":"low|medium|high", "reason":"short visual reason"}
```

## S1.5 Prompt D — OSR shadow-strength candidate review

```text
You are a visual-language reviewer for historical document shadow restoration. The panel shows the same page restored with five strengths: input, shadow_0.30, shadow_0.40, shadow_0.50, shadow_0.60. Use visible quality and the structured evidence to choose the safest useful restoration strength for a digital archive. Prefer a weaker strength when stronger candidates wash out paper texture, pale strokes, seals, annotations, or layout details. Prefer a stronger strength only when it clearly removes shadow without changing glyph strokes or layout. Return strict JSON only: {"choice":"input|shadow_0.30|shadow_0.40|shadow_0.50|shadow_0.60", "risk":"low|medium|high", "reason":"short visual reason"}. Structured evidence={STRUCTURED_EVIDENCE_JSON}
```

Expected JSON:

```json
{"choice":"input|shadow_0.30|shadow_0.40|shadow_0.50|shadow_0.60", "risk":"low|medium|high", "reason":"short visual reason"}
```

## S1.6 Prompt E — Hard-case pilot

```text
You are assisting a historical-document restoration pipeline. The image shows LEFT=input and RIGHT=a mild deshadow candidate. Use the visual evidence plus the structured metrics. Do not invent text. Choose one decision from: accept_candidate, keep_input, rollback_candidate, manual_review. Prefer content preservation over visual sharpness. Return strict JSON with keys decision, confidence, degradation, content_risk, reason. Dataset={DATASET}. Structured evidence={STRUCTURED_EVIDENCE_JSON}
```

Expected JSON:

```json
{"decision":"accept_candidate|keep_input|rollback_candidate|manual_review", "confidence":"number or short value", "degradation":"free-form or normalized degradation label", "content_risk":"low|medium|high", "reason":"short visual reason"}
```

## S1.7 Source-candidate evidence and VCCRP

The released implementation uses

```text
V_k = 0.45(1 - J_edge(x, y_k))
    + 0.25 rho_4(Delta_fg,k)
    + 0.20 rho_3(Delta_sigma,k)
    + 0.10 rho_4(Delta_mu,k)

rho_t(a) = min(t a, 1).
```

`J_edge` is edge-map Jaccard overlap, `Delta_fg` is normalized foreground-support shift, `Delta_sigma` is relative contrast change, and `Delta_mu` is normalized mean-intensity change.
