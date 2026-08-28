# Data and Code Availability

## Code Availability

This repository provides the public implementation components used to audit the manuscript results:

- candidate-bank definitions;
- source-relative evidence extraction, including VCCRP;
- region-aware candidate synthesis;
- semantic-prior calibration utilities;
- utility-based candidate selection;
- paired statistical analysis helpers;
- released split manifests and aggregate result tables.

Private server launch scripts, credentials, local absolute paths, raw datasets, third-party model weights, downloaded archives, and manuscript drafts are excluded.

## Dataset Availability

Raw image datasets are not redistributed in this repository. They should be obtained from the original providers and used under their respective licenses. The repository provides split manifests and result tables for auditability.

Datasets and sources used in the manuscript include:

- MixedDoc / MMDIR benchmark, obtained from the original MMDIR project or dataset provider.
- Historical-537, a grouped evaluation set assembled from the historical-document sources described in the manuscript.
- OSR, MTHv2, MACR, Kligler, Jung, and DIBCO subsets, obtained from their original sources or access channels.
- `compound-proxy`, a synthetic compound-degradation subset generated from clean historical source pages using the protocol in `configs/compound_proxy_generation_config.json` and `scripts/generate_compound_proxy.py`.

## Split Manifests

Provided split files:

- `splits/mixeddoc_stable_hash_split_manifest.csv`
- `splits/mixeddoc_stable_hash_split_counts.csv`
- `splits/historical537_group_split_manifest.csv`
- `splits/historical537_group_split_counts.csv`
- `splits/historical537_group_split_protocol.json`
- `splits/osr_split_manifest.csv`

The MixedDoc split follows:

```text
fnv1a32(sample_id) % 10:
  0-5 -> train
  6-7 -> val
  8-9 -> test
```

The Historical-537 split follows source-page grouping:

```text
md5(dataset + ":" + source_page_id) % 5 == 0 -> held-out validation
otherwise -> training
```

## Result Tables

Final result CSV files are included under `results/`. These files are sufficient to audit the reported aggregate scores and paired-statistical claims without redistributing original images or third-party model predictions.
