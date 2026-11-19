# delight_fm_v1

Final DELIGHT-FM training, downstream evaluation and in silico perturbation workflows.

Code: [ChloeHJ/delight_fm_v1](https://github.com/ChloeHJ/delight_fm_v1). Models: [chloehlee on Hugging Face](https://huggingface.co/chloehlee/). Input dataset destination: [chloehlee/delight_fm_v1-data](https://huggingface.co/datasets/chloehlee/delight_fm_v1-data) (publication pending).

## Contents

- `src/`: training, evaluation, perturbation, clustering and plotting code; environment requirements and tests.
- `runs/`: frozen configuration, train/test donor assignments, small regulatory reference tables and launchers.
- `docs/dataset/`: dataset card and portable SHA-256 manifest.
- `tools/dataset.py`: download, verification and explicit upload helper.

Large expression data and model weights are distributed on Hugging Face. This Git repository contains all scripts and small resources required to run the workflows after installing the input dataset. There are no runtime dependencies on historical v11 folders.

## Install data

Install `huggingface_hub` in a download environment. Once the dataset is published, use its exact 40-character commit SHA:

```bash
python tools/dataset.py download --revision COMMIT_SHA
```

The default dataset ID is `chloehlee/delight_fm_v1-data`. Downloads go to `runs` and every file is SHA-256 verified before the local receipt is written. See the [dataset card](docs/dataset/README.md).

In the local two-folder release, existing verified data live in adjacent `../huggingface/dataset`. Launchers detect that directory automatically, without copying the data. Set `DELIGHT_FM_RUN_DIR` to explicitly select another prepared run. Do not move a run while its jobs are active.

## Train and analyze

Install the Python 3.10/CUDA 11.8 modeling stack from `src/requirements-model.txt`; reporting dependencies are in `requirements-report.txt`. Cluster modules are used when available. Set the modeling interpreter explicitly:

```bash
export PYTHON_BIN=/path/to/model-environment/bin/python
bash runs/run.sh plan
bash runs/run.sh validate
bash runs/run.sh submit-training --dry-run
```

See [full workflow instructions](runs/README.md) and [source overview](src/README.md). The active run directory contains all inputs, configuration and outputs. For the local adjacent-data layout, edit configuration under `../huggingface/dataset/config`; `runs/config` is the distributable template.

Merged-model perturbations evaluate cell-type classification. Lineage models predict log2(PCW) and share one saved model and verified baseline across all genes and SCENIC+ regulons. Inference uses corrected float32 signed edits and mandatory GPU no-op controls. Clustering uses balanced magnitude-aware Ward with 12 groups. BHLHE40 uses bulk-informed proportional deletion and overexpression.

## Released models

Five foundation models: [merged](https://huggingface.co/chloehlee/delight-fm), [ileum](https://huggingface.co/chloehlee/delight-fm-epi-ileum), [colon](https://huggingface.co/chloehlee/delight-fm-epi-colon), [immune](https://huggingface.co/chloehlee/delight-fm-immune), [stromal](https://huggingface.co/chloehlee/delight-fm-stromal). The four lineage repositories have corresponding `-age` companions containing fitted age heads and their fine-tuned backbones. Their standalone model code, preprocessing resources and notebooks are packaged with the weights. Pin model commit revisions for reproducibility.

## Publication

Upload only this directory's Git-tracked contents to GitHub. `.gitignore` excludes inputs, weights, generated outputs and local credentials. The separate Hugging Face folder provides model upload instructions. The dataset helper can also upload a prepared run explicitly:

```bash
python tools/dataset.py upload --run-dir /path/to/prepared/run
```

New dataset repositories default to private. No upload occurs during preparation or verification. Complete dataset citation/redistribution terms and select the code-repository license before public release. Existing model packages retain their license and attribution files.
