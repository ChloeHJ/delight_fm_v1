---
pretty_name: DELIGHT-FM v1 reproducibility inputs
tags:
- single-cell
- gene-expression
- biology
---
# DELIGHT-FM v1 inputs

This dataset package contains 137 checksum-pinned files (161.94 GiB): five tokenized train/test datasets, matched expression H5AD and Matrix Market exports, metadata, gene statistics and source Seurat RDS files. `manifest.json` records relative paths, byte sizes, cohort and SHA-256. Files retain the directory layout expected by `runs` in the `delight_fm_v1` code repository.

The five cohorts are merged DELIGHT-FM, immune, stromal, colon epithelium and ileum epithelium. Frozen donor assignments and supporting regulatory reference tables accompany the code repository. Training and evaluation use separate donors. Colon retains the two recorded one-year samples in the <1-year category and places the teenager donor in training; ileum excludes the 80+ group. See the code repository run README for exact counts and selection methodology.

## Install

Install `huggingface_hub` in the download environment. From the code repository root, replace the namespace and revision with the published dataset repository and its commit SHA:

```bash
python tools/dataset.py download --repo-id chloehlee/delight_fm_v1-data --revision COMMIT_SHA
```

This downloads into `runs`, verifies every SHA-256, and generates a local input receipt. The datasets retain Hugging Face `save_to_disk` format and can be read with `datasets.load_from_disk`; they are not Parquet viewer tables. RDS and Matrix Market files support optional rebuilding and are redundant with prepared inputs for standard training.

## Release metadata

Dataset publication, citation and redistribution terms must be supplied by the dataset owner before public release. Model-package licenses do not determine the licensing of the underlying human expression data. This card describes locally staged research inputs; it does not claim that a public dataset repository already exists.

## Code and publication destination

Full training and analysis scripts, fixed donor assignments and regulatory references: [GitHub: ChloeHJ/delight_fm_v1](https://github.com/ChloeHJ/delight_fm_v1). Upload destination: [chloehlee/delight_fm_v1-data](https://huggingface.co/datasets/chloehlee/delight_fm_v1-data). The dataset helper uploads only the 137 manifest-listed inputs plus this card and manifest; local diagnostics, launchers and receipts are not uploaded. Download through the GitHub helper to install inputs into its prepared run layout.
