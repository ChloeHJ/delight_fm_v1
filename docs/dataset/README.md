---
pretty_name: DELIGHT-FM v1 data
# license: ""
tags:
- single-cell
- gene-expression
- foundation-model
---
# DELIGHT-FM v1 data

This dataset package contains 137 files (161.94 GiB): source Seurat RDS files, matched expression H5AD and Matrix Market exports, five tokenized train/test datasets, metadata, and gene statistics. `manifest.json` records relative paths, byte sizes, cohort and SHA-256. Files retain the directory layout expected by `runs` in the [delight_fm_v1](https://github.com/ChloeHJ/delight_fm_v1/) code repository.

The five cohorts are merged DELIGHT-FM, immune, stromal, colon epithelium and ileum epithelium. Frozen donor assignments and supporting regulatory reference tables accompany the code repository. Training and evaluation use separate donors. See the code repository README for exact counts and selection methodology.

## Install

The dataset is available at [chloehlee/delight_fm_v1-data](https://huggingface.co/datasets/chloehlee/delight_fm_v1-data). Download and verify the inputs with the helper in [ChloeHJ/delight_fm_v1](https://github.com/ChloeHJ/delight_fm_v1/):

```bash
git clone https://github.com/ChloeHJ/delight_fm_v1.git
cd delight_fm_v1
python -m pip install huggingface_hub
```

Replace `DATASET_COMMIT_SHA` below with the dataset's full 40-character commit SHA, available in the [dataset commit history](https://huggingface.co/datasets/chloehlee/delight_fm_v1-data/commits/main). Open a commit and copy the full 40-character SHA from the end of its URL. Pin the revision used for your analysis so the downloaded data and checksum manifest are reproducible.

```bash
python tools/dataset.py download \
  --repo-id chloehlee/delight_fm_v1-data \
  --revision DATASET_COMMIT_SHA \
  --run-dir runs
```

The helper downloads the 137 manifest-listed files (approximately 161.94 GiB). Input paths resolve relative to `runs/`, training and analysis configuration in the GitHub repository. See its [run instructions](https://github.com/ChloeHJ/delight_fm_v1/blob/main/runs/README.md) for the modeling environment and workflow commands.

The tokenized train/test datasets retain Hugging Face `save_to_disk` format and can be opened with `datasets.load_from_disk` in the modeling environment. The Seurat RDS and Matrix Market files support optional expression export and rebuilding; the prepared tokenized datasets are used for the standard training workflow.

## Release metadata

| Field | Information to complete |
|---|---|
| Dataset release version | `""` |
| Release date | `""` |
| Authors / data contributors | `""` |
| Maintainer and contact | `""` |
| Dataset DOI or persistent identifier | `""` |
| Associated publication / preprint | `""` |
| Dataset license | `""` |
| Permitted uses and access conditions | `""` |
| Redistribution terms | `""` |
| Required attribution | `""` |


### Source datasets and terms

| Study / source | Accession or source URL | Source citation | License / terms | Attribution or redistribution conditions |
|---|---|---|---|---|
| `""` | `""` | `""` | `""` | `""` |

### Citation

`""`

Fill the empty BibTeX fields below with the final citation details. Add the associated publication citation when available.

```bibtex
@misc{delight_fm_v1_data,
  author       = "",
  title        = {DELIGHT-FM v1 reproducibility inputs},
  year         = "",
  version      = "",
  doi          = "",
  url          = {https://huggingface.co/datasets/chloehlee/delight_fm_v1-data}
}
```

## Code and data

Full training and analysis scripts, fixed donor assignments and regulatory references: [GitHub: ChloeHJ/delight_fm_v1](https://github.com/ChloeHJ/delight_fm_v1). Published dataset: [chloehlee/delight_fm_v1-data](https://huggingface.co/datasets/chloehlee/delight_fm_v1-data).
