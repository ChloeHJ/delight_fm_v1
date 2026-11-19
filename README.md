# DELIGHT-FM v1.0: a foundation model for intestinal maturation

DELIGHT-FM training, downstream evaluation and in silico perturbation workflows.

Code: [ChloeHJ/delight_fm_v1](https://github.com/ChloeHJ/delight_fm_v1). Models: [chloehlee on Hugging Face](https://huggingface.co/chloehlee/). Published input dataset: [chloehlee/delight_fm_v1-data](https://huggingface.co/datasets/chloehlee/delight_fm_v1-data).

## Contents

- `src/`: training, evaluation, perturbation, clustering and plotting code; environment requirements and tests.
- `runs/`: frozen configuration, train/test donor assignments, small regulatory reference tables and launchers.
- `docs/dataset/`: dataset card and portable SHA-256 manifest.
- `tools/dataset.py`: download, verification and explicit upload helper.

Large expression data and model weights are distributed on Hugging Face. This Git repository contains all scripts and small resources required to run the workflows after installing the input dataset. All required workflow code is included in this repository.

## Install data

Install `huggingface_hub` in a download environment. Use the published dataset’s full 40-character commit SHA from its [commit history](https://huggingface.co/datasets/chloehlee/delight_fm_v1-data/commits/main):

```bash
python tools/dataset.py download --revision COMMIT_SHA
```

The default dataset ID is `chloehlee/delight_fm_v1-data`. Downloads go to `runs` and every file is SHA-256 verified before the local receipt is written. See the [dataset card](docs/dataset/README.md).

In the local two-folder release, existing verified data live in adjacent `../huggingface/dataset`. Launchers detect that directory automatically, without copying the data. Set `DELIGHT_FM_RUN_DIR` to explicitly select another prepared run. Do not move a run while its jobs are active.

## Train and analyze

Install the Python 3.10/CUDA 11.8 modeling stack from `src/requirements-model.txt`; reporting dependencies are in `src/requirements-report.txt`. Cluster modules are used when available. Set the modeling interpreter explicitly:

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

Saved age models: [ileum age](https://huggingface.co/chloehlee/delight-fm-epi-ileum-age), [colon age](https://huggingface.co/chloehlee/delight-fm-epi-colon-age), [immune age](https://huggingface.co/chloehlee/delight-fm-immune-age), and [stromal age](https://huggingface.co/chloehlee/delight-fm-stromal-age). Each model repository includes its usage notebooks.

See [model task support and notebook guide](docs/MODEL_TASKS.md) for saved-head inference, task-specific fine-tuning, lineage age prediction, SCENIC+ age perturbations and merged CollecTRI cell-type perturbations.

## Citation

Postnatal Reprogramming Shapes Human Intestinal Epithelial Immune Competency

Chloe Hyun-Jung Lee, David Fawkner-Corbett, Zoe Christoforidou, Ana Sousa Gerós, Verena Lentsch, Lucy Sheikh, Esther Bridges, Marta Jagielowicz, Lei Deng, Xiao Qin, Huei-Wen Chuang, Vy Wien Lai, Sophie Craddock, Aurore Mazurier, Paulina Siejka-Zielinska, Paula Gomez Castro, Anna Aulicino, Colleen McGregor, Tarun Gupta, Nicole Cianci, Rhea Kujawa, Paola Vargas Gutierrez, Carol Cheng, Maria Greco, Darren Fowler, Simon James Alexander Buczacki, Grace Rimmer, Rachel Harwood, Nigel Hall, Paul Johnson, Hashem Koohy, Alison Simmons, Agne Antanaviciute

bioRxiv 2026.05.05.722861; doi: https://doi.org/10.64898/2026.05.05.722861