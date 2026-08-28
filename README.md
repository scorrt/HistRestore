# HistRestore

Code and result artifacts for **HistRestore: evidence-constrained restoration selection with vision-language priors for historical document images**.

This public release is a reproducibility and audit package for the manuscript. It contains reusable HistRestore modules, split manifests, candidate-bank configurations, synthetic-degradation generation code, and final result tables. It does **not** include original datasets, third-party model weights, private server scripts, downloaded benchmark archives, or generated manuscript drafts.

## What Is Included

- `scripts/`: public scripts for compound-proxy generation and released-result summarization.
- `src/histrestore/`: reusable project code for candidate-bank construction, evidence extraction, region-aware candidate synthesis, distilled semantic priors, utility selection, and paired statistical tests.
- `results/`: final CSV summaries, paired statistics, runtime summaries, and audit reports.
- `splits/`: group split and stable-hash split manifests.
- `configs/`: candidate-bank and compound-proxy generation definitions.
- `docs/`: data availability and reproducibility notes.

The `src/histrestore/` package contains the method-level implementation used by the release. Some end-to-end benchmark runs require third-party datasets, model checkpoints, official predictions, or Qwen review logs that cannot be redistributed here; the released CSV tables are provided to audit the aggregate manuscript results.

## Main Reproducibility Targets

The key paper-ready MixedDoc protocol is:

- Candidate bank: `input_degraded`, classical shadow correction, DocRes variants, region candidate, and `mmdir_official`.
- Split: stable hash 60/20/20 over 1837 MixedDoc pages.
- Test set: 377 pages.
- Main result file: `results/mixedoc/main_sota_table_mixeddoc.csv`.

Main MixedDoc test result:

| Method | PSNR | SSIM | VCCRP |
|---|---:|---:|---:|
| MMDIR official | 27.5196 | 0.9479 | 0.4364 |
| HistRestore + MMDIR | 27.6673 | 0.9480 | 0.4306 |
| Oracle candidate pool | 27.8588 | 0.9459 | 0.4231 |

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Optional dependencies are needed only for specific components:

- `opencv-python` and `scikit-image` for image metrics.
- `scikit-learn` and `scipy` for selectors and statistical tests.
- `pandas` for table generation.

## Data Preparation

This repository does not redistribute datasets or model weights. Download datasets from their original providers and place regenerated artifacts in a local working directory following the policies described in `docs/DATA_AVAILABILITY.md`.

For manuscript verification without raw images, use the provided CSV result tables in `results/`.

## Auditing Released Tables

The release includes final CSV summaries. To print the main released tables:

```bash
python scripts/summarize_released_results.py
```

To regenerate the compound-proxy subset from clean source pages:

```bash
python scripts/generate_compound_proxy.py --clean-root path/to/clean_pages --out-root path/to/compound_proxy
```

## Repository Hygiene

This release intentionally excludes:

- raw images and benchmark archives;
- third-party neural network checkpoints;
- server addresses, SSH helpers, and credentials;
- temporary exploratory outputs;
- generated manuscript drafts.

## Citation

If this code is used, please cite the associated manuscript after publication.
