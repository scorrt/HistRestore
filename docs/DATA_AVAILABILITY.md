# Data and Code Availability

## Code Availability

The sanitized code used for candidate-bank evaluation, evidence extraction, semantic-prior analysis, selector training, paired bootstrap statistics, and final table generation is included in this repository.

The repository excludes private server scripts, credentials, local absolute paths, raw datasets, and third-party model weights.

## Dataset Availability

The experiments use public or separately distributed document-image datasets. Due to licensing and redistribution restrictions, raw images are not included in this repository. Users should obtain each dataset from its original provider.

Datasets and sources used in the manuscript include:

- MixedDoc / MMDIR benchmark: obtain from the original MMDIR project or dataset provider.
- Historical-537: constructed as a grouped historical-document evaluation set from the sources described in the manuscript. The group split manifest is provided in `splits/historical537_group_split_manifest.csv`.
- OSR: external shadow-removal evaluation protocol; split manifest is provided when available.
- MTHv2, MACR, Kligler, Jung, DIBCO subsets: obtain from their original sources or according to the access conditions described in the manuscript.

## Split Manifests

Provided split files:

- `splits/mixeddoc_stable_hash_split_manifest.csv`
- `splits/mixeddoc_stable_hash_split_counts.csv`
- `splits/historical537_group_split_manifest.csv`
- `splits/historical537_group_split_counts.csv`
- `splits/historical537_group_split_protocol.json`

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

Final result CSV files are included under `results/`. These files are sufficient to audit the reported aggregate scores and paired-statistical claims without redistributing original images.

## Suggested Manuscript Statement

The following statement can be adapted for MDPI submission:

> The code for evidence extraction, candidate selection, semantic-prior analysis, and statistical evaluation is available at the project GitHub repository. The raw datasets and third-party restoration outputs are not redistributed due to licensing restrictions and should be obtained from their original providers. The repository includes split manifests and result tables needed to reproduce the reported aggregate analyses.

