# HistRestore

Companion code and released result artifacts for **HistRestore: Evidence-Constrained Restoration Selection with Vision-Language Priors for Historical Document Images**.

HistRestore treats document restoration as page-wise selection over an explicit candidate bank. Candidates are compared with the degraded source using structural, foreground, appearance, and preservation-risk evidence. Historical-537 evaluates direct structured Qwen3-VL review; MixedDoc evaluates train-only prior distillation for VLM-free test-time selection.

## Repository contents

- `src/histrestore/` — candidate-bank utilities, source-candidate evidence, VCCRP, region-aware candidate synthesis, semantic-prior features, utility selection, and paired statistics.
- `scripts/` — compound-proxy generation, released-result summaries, and release consistency checks.
- `configs/` — Historical-537 nested candidate order, MixedDoc candidate banks, compound-proxy parameters, and Qwen label mapping.
- `splits/` — final Historical-537, MixedDoc, and OSR split manifests.
- `results/` — Historical-537, MixedDoc, NR-IQA, selector-family, sensitivity, semantic, OCR, region-aware, runtime, and per-page audit files.
- `docs/` — Appendix S1 prompt templates, reproducibility notes, supplementary-file mapping, and data availability.

## Main released results

Historical-537 uses 424 training pages and 113 source-group-held-out pages. The 22-candidate oracle reaches 26.088 dB PSNR, the strongest fixed blend 24.576 dB, evidence-only HistRestore 24.683 dB, and direct structured review 25.006 dB.

MixedDoc uses a fixed 1083/377/377 train/validation/test partition. On the 377-page test set, the official MMDIR output reaches 27.520 dB PSNR and distilled-prior HistRestore reaches 27.667 dB, with VCCRP reduced from 0.4364 to 0.4306.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick checks

```bash
python scripts/summarize_released_results.py
python scripts/validate_release.py
```

## Compound-proxy generation

Historical-537 contains 120 compound-proxy samples generated from 60 clean Jung source pages, with two degraded variants per source page.

```bash
python scripts/generate_compound_proxy.py \
  --clean-root path/to/clean_pages \
  --out-root path/to/compound_proxy \
  --variants 2
```

The generator uses seed `20260805`, a maximum long side of 1600 pixels, one to three sampled degradation operators, severity in `[0.25, 0.90]`, and optional JPEG compression.

## Supplementary mapping

`docs/SUPPLEMENTARY_FILE_INDEX.md` maps the manuscript Supplementary Materials statement to the released machine-readable files. Prompt A–E and their JSON schemas are in `docs/Supplementary_Appendix_S1.md`.

## License

MIT License. Dataset and third-party model usage remains subject to the licenses of the original providers.
