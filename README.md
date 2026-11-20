# DELIGHT-FM v1.0: a foundation model for intestinal maturation

This repository provides the workflows for training DELIGHT-FM, evaluating its predictions and exploring how in silico gene and regulon perturbations affect predicted cell identity or developmental age. To use the released models directly, start with the [model task and notebook guide](docs/MODEL_TASKS.md). To train models and run the full analysis workflow, follow the setup below.

Code: [ChloeHJ/delight_fm_v1](https://github.com/ChloeHJ/delight_fm_v1). Models: [Hugging Face](https://huggingface.co/chloehlee/). Published input dataset: [chloehlee/delight_fm_v1-data](https://huggingface.co/datasets/chloehlee/delight_fm_v1-data).

## Contents

- `src/`: training, evaluation, perturbation, clustering and plotting code; environment requirements and tests.
- `runs/`: frozen configuration, train/test donor assignments, small regulatory reference tables and launchers.
- `docs/dataset/`: dataset card and portable SHA-256 manifest.
- `tools/dataset.py`: download, verification and explicit upload helper.

Large expression datasets and model weights are distributed on Hugging Face. This Git repository contains the workflow code, configuration and small reference resources; install the input dataset separately before running training and analysis.

## Install data

Run the following commands from the root of this repository. First, install the Hugging Face download helper in the Python environment you will use to download the data (you can skip this step if it is already installed):

```bash
python -m pip install huggingface_hub
```

Next, choose the dataset version to download. Open the published dataset’s [commit history](https://huggingface.co/datasets/chloehlee/delight_fm_v1-data/commits/main), open the desired commit and copy the full 40-character SHA from the end of its URL. Replace `COMMIT_SHA` below with that value:

```bash
python tools/dataset.py download --revision COMMIT_SHA
```

The commit SHA pins an exact dataset version, so you can retrieve the same inputs even if the dataset repository is updated later. The helper also checks every downloaded file against this code release’s SHA-256 manifest to confirm that the inputs match the expected files.

The default dataset is `chloehlee/delight_fm_v1-data`, containing 137 files (approximately 162 GiB). In a standalone checkout, downloads go to `runs/`. After successful verification, the helper writes a local receipt at `data/input_manifest.json` inside the run directory. See the [dataset card](docs/dataset/README.md) for the included files and download options.

If you are using the local two-folder release, verified data are already available in the adjacent `../huggingface/dataset` directory. The download helper and workflow launchers detect that prepared directory automatically, so you can use the existing data without downloading another copy. To select another prepared run for the workflow, set `DELIGHT_FM_RUN_DIR` to its path. Keep the run directory in place while its jobs are active.

## Train and analyze

Training and model inference use the Python 3.10/CUDA 11.8 environment described in [the modeling requirements](src/requirements-model.txt). Plotting and reports use a separate environment described in [the reporting requirements](src/requirements-report.txt). See the [source overview](src/README.md#environments-and-checks) for environment details, including matching the compiled Mamba extensions to PyTorch. The launcher loads cluster software modules when available.

Once the modeling environment is ready, replace the example path below with its Python executable. These commands freeze the run configuration, prepare and validate the inputs, and preview the Slurm training submission:

```bash
export PYTHON_BIN=/path/to/model-environment/bin/python
bash runs/run.sh plan
bash runs/run.sh prepare
bash runs/run.sh validate
bash runs/run.sh submit-training --dry-run
```

Review the preview before submitting jobs. The [full workflow instructions](runs/README.md) cover training submission, perturbation screens and report generation. The active run directory contains the inputs, configuration and outputs. Review its configuration before running `plan`; for the local adjacent-data layout, edit `../huggingface/dataset/config`. The `runs/config` directory provides the distributable configuration template.

The analysis workflows address several related questions:

- **Cell identity:** perturbations in the merged model measure changes in cell-type classification, using curated markers and CollecTRI regulons.
- **Developmental age:** the four lineage age models predict log2-transformed age in post-conception weeks (PCW). Within each lineage, all-gene and SCENIC+ regulon screens reuse one saved age model and the same verified baseline predictions, so their effects can be compared against a common reference.
- **Regulon response patterns:** when clustering regulons by their perturbation effects on predicted age across donors, the workflow uses balanced magnitude-aware Ward clustering with 12 groups by default. A shared scaling transformation preserves the direction and relative magnitude of the effects. Reports include representative regulons and checks of how stable the groups are when individual donors are left out.
- **BHLHE40:** the ileum workflow uses bulk-expression-informed proportional deletion and overexpression to examine changes in predicted age. The deletion applies the inverse of the bulk overexpression response; it does not represent a measured knockout signature.

Perturbation inference uses float32 edits and requires GPU no-op controls to pass before screens proceed. These controls check that edits intended to leave the input unchanged also leave predictions unchanged.

## Released models

Five foundation models are available: [merged](https://huggingface.co/chloehlee/delight-fm), [ileum](https://huggingface.co/chloehlee/delight-fm-epi-ileum), [colon](https://huggingface.co/chloehlee/delight-fm-epi-colon), [immune](https://huggingface.co/chloehlee/delight-fm-immune) and [stromal](https://huggingface.co/chloehlee/delight-fm-stromal). Each repository packages the weights with standalone model code, preprocessing resources and usage notebooks.

For developmental-age prediction and age perturbations, use the corresponding saved age models: [ileum age](https://huggingface.co/chloehlee/delight-fm-epi-ileum-age), [colon age](https://huggingface.co/chloehlee/delight-fm-epi-colon-age), [immune age](https://huggingface.co/chloehlee/delight-fm-immune-age) and [stromal age](https://huggingface.co/chloehlee/delight-fm-stromal-age). These `-age` repositories contain fitted age heads, their fine-tuned backbones and usage notebooks. Pin model commit revisions as well as the dataset revision to reproduce an analysis; each repository has its own commit SHA.

See [model task support and notebook guide](docs/MODEL_TASKS.md) for saved-head inference, task-specific fine-tuning, lineage age prediction, SCENIC+ age perturbations and merged CollecTRI cell-type perturbations.

## Citation

Postnatal Reprogramming Shapes Human Intestinal Epithelial Immune Competency

Chloe Hyun-Jung Lee, David Fawkner-Corbett, Zoe Christoforidou, Ana Sousa Gerós, Verena Lentsch, Lucy Sheikh, Esther Bridges, Marta Jagielowicz, Lei Deng, Xiao Qin, Huei-Wen Chuang, Vy Wien Lai, Sophie Craddock, Aurore Mazurier, Paulina Siejka-Zielinska, Paula Gomez Castro, Anna Aulicino, Colleen McGregor, Tarun Gupta, Nicole Cianci, Rhea Kujawa, Paola Vargas Gutierrez, Carol Cheng, Maria Greco, Darren Fowler, Simon James Alexander Buczacki, Grace Rimmer, Rachel Harwood, Nigel Hall, Paul Johnson, Hashem Koohy, Alison Simmons, Agne Antanaviciute

bioRxiv 2026.05.05.722861; doi: https://doi.org/10.64898/2026.05.05.722861
