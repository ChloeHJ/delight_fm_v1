# DELIGHT-FM v1.0: a foundation model for intestinal maturation

This repository contains the scripts needed to reproduce DELIGHT-FM model development and downstream analyses: starting from Seurat RDS files, exporting expression data and metadata, tokenising cells, pretraining models, fine-tuning and evaluating task-specific models, and applying them to in silico gene and regulon perturbations of predicted cell identity or developmental age. Prepared tokenised datasets are also provided so you can start training without rebuilding the inputs.

To use the released models directly, start with the [model task and notebook guide](docs/MODEL_TASKS.md). To prepare inputs, train models and run the analysis workflows, follow the setup below and the [full workflow instructions](runs/README.md).

Code: [ChloeHJ/delight_fm_v1](https://github.com/ChloeHJ/delight_fm_v1). Models: [Hugging Face](https://huggingface.co/chloehlee/). Published input dataset: [chloehlee/delight_fm_v1-data](https://huggingface.co/datasets/chloehlee/delight_fm_v1-data).

## Contents

- `src/`: training, evaluation, perturbation, clustering and plotting code; environment requirements and tests.
- `runs/`: frozen configuration, train/test donor assignments, small regulatory reference tables and launchers.
- `docs/dataset/`: dataset card and portable SHA-256 manifest.
- `tools/dataset.py`: download, verification and explicit upload helper.

Large expression datasets and model weights are distributed on Hugging Face. This Git repository contains the workflow code, configuration and small reference resources; install the input dataset separately before running training and analysis.

## Set up the environments

Run these commands from the root of this repository. The workflow uses two Python environments: one for training and model inference, and one for plotting and reports. The examples below create them under `$HOME/.venvs` without requiring administrator access.

### 1. Create the modeling environment

Use Linux with Python 3.10 and the CUDA 11.8 toolkit available. On a cluster with environment modules, load CUDA first:

```bash
module load cuda/11.8
```

If your system does not use modules, make sure CUDA 11.8 is on your executable and library paths. Check that `python3.10 --version` and `nvcc --version` report the expected versions. Building the CUDA extensions from source also requires a compatible C++ compiler.

```bash
python3.10 -m venv "$HOME/.venvs/delight-fm-model"
export PYTHON_BIN="$HOME/.venvs/delight-fm-model/bin/python"

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel packaging ninja
"$PYTHON_BIN" -m pip install numpy==1.26.4
"$PYTHON_BIN" -m pip install torch==2.2.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
"$PYTHON_BIN" -m pip install --no-build-isolation -r src/requirements-model.txt
```

Install PyTorch first so the [Mamba](https://github.com/state-spaces/mamba/blob/v2.2.2/setup.py) and [causal-convolution](https://github.com/Dao-AILab/causal-conv1d/blob/v1.4.0/setup.py) extensions can use the matching PyTorch/CUDA environment. `--no-build-isolation` lets their build steps access those installed dependencies. The pinned package versions are recorded in [the modeling requirements](src/requirements-model.txt); the CUDA wheel index follows the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v222).

The same environment can download the dataset:

```bash
"$PYTHON_BIN" -m pip install huggingface_hub
```

### 2. Create the reporting environment

Reporting and regulon clustering can run on a CPU using a separate environment:

```bash
python3.10 -m venv "$HOME/.venvs/delight-fm-report"
export REPORT_PYTHON_BIN="$HOME/.venvs/delight-fm-report/bin/python"
"$REPORT_PYTHON_BIN" -m pip install --upgrade pip
"$REPORT_PYTHON_BIN" -m pip install -r src/requirements-report.txt
```

The launcher uses `REPORT_PYTHON_BIN` for `report` and `cluster`. If you leave it unset on a cluster with modules, the launcher loads `python-cbrg` instead.

### 3. Check the setup

```bash
"$PYTHON_BIN" -m pip check
env -u PYTHONPATH PYTHONNOUSERSITE=1 "$PYTHON_BIN" - <<'PY'
import torch
import transformers, datasets, mamba_ssm, causal_conv1d
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
PY
"$REPORT_PYTHON_BIN" -c 'import numpy, pandas, scipy, sklearn, matplotlib, adjustText; print("Reporting imports OK")'
```

Expect PyTorch `2.2.2+cu118` and CUDA `11.8`. Run the GPU availability check on a node with an allocated NVIDIA GPU; a login or CPU-only node can report `False`. The workflow runs its full baseline and no-op controls before perturbation screens.

In a new shell, restore the interpreter paths before running the workflow (and load CUDA again if your system requires it):

```bash
export PYTHON_BIN="$HOME/.venvs/delight-fm-model/bin/python"
export REPORT_PYTHON_BIN="$HOME/.venvs/delight-fm-report/bin/python"
```

Training submission uses Slurm (`sbatch`). Review the partition names and resources in the active run’s `config/settings.json` before submitting jobs. R is needed only for the optional RNA input rebuild; the prepared input datasets can be used directly.

## Install data

Run the following commands from the root of this repository. First, install the Hugging Face download helper in the Python environment you will use to download the data (you can skip this step if it is already installed):

```bash
python -m pip install huggingface_hub
```

If you created the modeling environment above, use `"$PYTHON_BIN"` in place of `python` in the commands in this section.

Next, choose the dataset version to download. Open the published dataset’s [commit history](https://huggingface.co/datasets/chloehlee/delight_fm_v1-data/commits/main), open the desired commit and copy the full 40-character SHA from the end of its URL. Replace `COMMIT_SHA` below with that value:

```bash
python tools/dataset.py download --revision COMMIT_SHA
```

The commit SHA pins an exact dataset version, so you can retrieve the same inputs even if the dataset repository is updated later. The helper also checks every downloaded file against this code release’s SHA-256 manifest to confirm that the inputs match the expected files.

The default dataset is `chloehlee/delight_fm_v1-data`, containing 137 files (approximately 162 GiB). In a standalone checkout, downloads go to `runs/`. After successful verification, the helper writes a local receipt at `data/input_manifest.json` inside the run directory. See the [dataset card](docs/dataset/README.md) for the included files and download options.

If you already have the dataset stored at `../huggingface/dataset` relative to this repository, the workflow automatically uses that directory. Otherwise, download the dataset using the command above.

To select another prepared run for the workflow, set `DELIGHT_FM_RUN_DIR` to its path. Keep the run directory in place while its jobs are active.

## Train and analyze

The workflow below trains new models, evaluates candidate checkpoints and verifies the saved models before running perturbation analyses. To work directly with published checkpoints instead, use the [model task and notebook guide](docs/MODEL_TASKS.md).

### 1. Configure the run and validate the inputs

Install the dataset first and set `PYTHON_BIN` and `REPORT_PYTHON_BIN` as described in [the environment setup](#set-up-the-environments). The active run directory contains the inputs, configuration and outputs. By default, it is `runs/`, or `../huggingface/dataset` when that prepared dataset directory is detected.

Before running `plan`, review the active run’s `config/settings.json` for the training configurations, checkpoint epochs and Slurm resources, and `config/inputs.json` for input paths. If you already have data in `../huggingface/dataset`, edit the configuration there. The `runs/config` directory provides the distributable template.

```bash
bash runs/run.sh plan
bash runs/run.sh prepare
bash runs/run.sh validate
```

`plan` records the configuration and input/source hashes so subsequent stages use a consistent run. `prepare` resolves the prepared input directories and validates them; `validate` can also be run separately to repeat those checks. Input validation checks donor assignments, absence of train/test donor overlap, cell counts, age labels, sequence lengths and required age-group coverage. It writes cohort summaries and age counts under `diagnostics/`. These checks validate the data layout and splits; model performance is evaluated in the next stage.

### 2. Train and evaluate candidate models

Preview the Slurm jobs, then submit the training workflow after reviewing the commands:

```bash
bash runs/run.sh submit-training --dry-run
bash runs/run.sh submit-training
```

Submission schedules the following stages with dependencies, so each starts after the preceding jobs succeed:

1. **Prepare inputs:** repeat input preparation and validation on the scheduled jobs.
2. **Pretrain models:** train the configured merged and lineage model variants. Defaults are 22 configurations trained for 10 epochs, with checkpoints saved every half epoch.
3. **Evaluate checkpoints:** fine-tune each candidate for cell-type, cluster and location classification, and developmental-age regression. Age evaluation uses five-fold donor-grouped cross-validation within the training donors, plus evaluation on the fixed held-out donors. The default sweep evaluates 20 checkpoints per configuration (440 checkpoint evaluation jobs).
4. **Select a checkpoint per cohort:** rank candidates by the mean of cluster macro-F1, location macro-F1 and held-out age Pearson correlation. Lower pretraining validation loss breaks ties; broad-cell-type scores are reported separately. A classification task with only one training class is marked as not estimable.
5. **Prepare analysis models and baselines:** fit and save one final age regression model per lineage, reuse the selected merged classifier, and verify baseline predictions and GPU no-op controls before perturbations.

The held-out donors are used for checkpoint selection, so their reported performance describes the selection set. An independent final test requires an additional donor cohort that was not used for selection.

### 3. Review training and validation results

Wait for the training, evaluation and baseline jobs to finish successfully before submitting perturbation screens. Check job logs under `logs/` and the submission record at `diagnostics/submission_training.json` in the active run. The submission command returns after scheduling jobs; it does not wait for training to finish.

Generate the training and evaluation reports with:

```bash
bash runs/run.sh report
```

Review the training loss curves, downstream metrics and model-selection results. The main report is `outputs/web_summary/index.html`; rankings and selected checkpoints are under `outputs/model_selection/`. Each cohort’s `outputs/model_registry/<group>/` contains the saved-model binding, `noop_controls.json` and `baseline_verified.json`. These records establish which model, inputs and unperturbed predictions the analysis will use. Perturbation submission checks that the models and baselines still match the verified records.

### 4. Run perturbation analyses

Once all five model baselines are verified, preview and submit the gene and regulon screens:

```bash
bash runs/run.sh submit-perturbations --dry-run
bash runs/run.sh submit-perturbations
```

This schedules the merged marker/CollecTRI screens and the four lineage all-gene/SCENIC+ age screens, followed by merging and report generation. After the regulon results have finished merging, group their response patterns and refresh the reports:

```bash
bash runs/run.sh cluster
bash runs/run.sh report
```

The separate BHLHE40 analysis can be submitted after the ileum age model and baseline are verified:

```bash
bash runs/run.sh submit-bhlhe40 --dry-run
bash runs/run.sh submit-bhlhe40
```

See the [full workflow instructions](runs/README.md) for individual stages, recovery commands and output locations.

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
