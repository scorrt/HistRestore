# HistRestore

Code and result artifacts for **HistRestore: an evidence-constrained adaptive restoration framework for historical document images**.

This public release is a reproducibility package for the manuscript. It contains the reusable HistRestore implementation modules, sanitized experiment scripts, split manifests, candidate-bank configurations, and final result tables. It does **not** include original datasets, third-party model weights, private server scripts, or downloaded benchmark archives.

## What Is Included

- `scripts/`: core evaluation and analysis scripts used for the paper tables.
- `src/histrestore/`: reusable project code for candidate-bank construction, evidence extraction, region-aware candidate synthesis, distilled semantic priors, utility selection, and paired statistical tests.
- `results/`: final CSV summaries, paired statistics, runtime summaries, and audit reports.
- `splits/`: group split and stable-hash split manifests.
- `configs/`: candidate-bank definitions.
- `docs/`: data availability, reproducibility notes, and upload instructions.

The `src/histrestore/` package contains the method-level implementation skeleton, while `scripts/` contains paper-table entrypoints that assemble the released CSVs and precomputed candidate metrics into the reported protocols.

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

This repository does not redistribute datasets or model weights. Download datasets from their original providers and place generated metric files under a local `data/` or `outputs/` directory following the paths described in `docs/DATA_AVAILABILITY.md`.

For manuscript verification without raw images, use the provided CSV result tables in `results/`.

## Reproducing Reported Tables

Examples:

```bash
python scripts/build_mmdir_frozen_compat_balanced_20260827.py
python scripts/build_frozen_compat_external_20260827.py
```

Some scripts expect precomputed candidate metrics and official predictions. These files are not redistributed when their source datasets or model outputs are subject to third-party licensing.

## Repository Hygiene

This release intentionally excludes:

- raw images and benchmark archives;
- third-party neural network checkpoints;
- server addresses, SSH helpers, and credentials;
- temporary exploratory outputs;
- generated manuscript drafts.

## Citation

If this code is used, please cite the associated manuscript after publication.
