# Run DELIGHT-FM v1

This guide provides the detailed commands for reproducing DELIGHT-FM training, checkpoint evaluation and downstream analyses. The workflow starts with prepared tokenised datasets, trains and selects models, verifies their baseline predictions, and then applies gene and regulon perturbations. An optional input rebuild starts from Seurat RDS files. To use the released models directly, follow the [model task and notebook guide](../docs/MODEL_TASKS.md).

Run all commands from the repository root. First complete the [environment setup](../README.md#set-up-the-environments) and [data installation](../README.md#install-data) in the main README. Set `PYTHON_BIN` for training and inference, and `REPORT_PYTHON_BIN` for plotting and clustering. If you used the example environment locations, restore these paths in each new shell:

```bash
export PYTHON_BIN="$HOME/.venvs/delight-fm-model/bin/python"
export REPORT_PYTHON_BIN="$HOME/.venvs/delight-fm-report/bin/python"
```

The active run directory contains the inputs, configuration, donor assignments and outputs. It defaults to `runs/`, or `../huggingface/dataset` when that prepared dataset directory is detected. Set `DELIGHT_FM_RUN_DIR` to select another prepared run. Paths below are relative to the active run unless shown as repository commands; edit its configuration rather than the `runs/config` template when using another run directory.

The launchers do not load environment modules. Prepare CUDA and any site-specific modules before running them. Optional RNA rebuilding uses `RSCRIPT_BIN` (default: `Rscript` on PATH). Scheduled jobs use the launchers from the current Git checkout, with data and outputs in the selected run directory.

The `submit-*` commands require Slurm. Set `slurm.gpu_partition` and `slurm.cpu_partition` in `config/settings.json` for your cluster; `null` uses the scheduler default without adding a partition argument. Review all resource settings before submission. To use another scheduler or run directly, invoke individual stages with the appropriate task IDs and allocate the required CPU/GPU resources yourself.

## Cohorts

| Model | Train cells | Test cells | Train donors | Test donors |
|---|---:|---:|---:|---:|
| Merged DELIGHT-FM | 531,101 | 140,257 | 128 | 21 |
| Immune | 175,031 | 36,586 | 126 | 21 |
| Colon epithelium | 85,035 | 25,155 | 79 | 13 |
| Ileum epithelium | 67,849 | 14,564 | 71 | 13 |
| Stromal | 206,814 | 60,323 | 119 | 21 |

The released tokenised datasets use fixed donor assignments with no train/test donor overlap. Ileum excludes the `Adult, 80+` age group; colon and ileum each require at least 50 training cells in every retained age group. Each cohort contains its available subset of donors, so donor sets differ across lineages. Validation exports per-age cell and donor counts to `diagnostics/*_age_counts.csv`.

## 1. Configure the run and validate the inputs

Before planning, review `config/settings.json` for the hyperparameter grids, checkpoint epochs, marker/TF panels and Slurm resources, and `config/inputs.json` for input paths. Defaults are 6 merged configurations and 4 per lineage (22 training jobs), with 20 half-epoch evaluations from 0.5 to 10 epochs (440 checkpoint evaluation jobs). CollecTRI defaults to a configured 13-TF panel; set `perturbation.collectri_tfs` to `"all"` before planning to use the full bundled human table. Lineage regulons use the bundled SCENIC+ tables.

```bash
bash runs/run.sh plan
bash runs/run.sh prepare
bash runs/run.sh validate
```

`plan` records configuration and input/source hashes in `diagnostics/plan.json` and writes the training task IDs to `conditions.tsv`. `prepare` resolves the prepared input directories and validates them. `validate` repeats the checks independently: donor assignments, absence of train/test donor overlap, cell counts, age labels, sequence lengths and required age-group coverage. Summaries are written to `diagnostics/`. Input validation checks the data and splits; model performance is evaluated after training.

The plan is immutable and can be repeated unchanged. Computational changes require a fresh run/plan. Slurm resources and clustering resolution can change without invalidating fitted models. To start another analysis, prepare a new directory with the configuration, references, donors and verified local inputs, set `DELIGHT_FM_RUN_DIR` to its path, and create a plan there. Keep generated results and diagnostics with their original run, and keep run directories in place while jobs are active.

For the optional Seurat RDS export and tokenisation route, use `bash runs/run.sh rebuild-inputs --group GROUP` in a fresh run with empty exported/tokenised destinations. This requires R with Seurat (v5 API), Matrix and jsonlite; set `RSCRIPT_BIN` if `Rscript` is not on your PATH. Use a group name from the [recovery commands](#individual-stages-and-recovery) below. The standard training workflow uses the supplied tokenised datasets and does not need R.

## 2. Train and evaluate candidate models

Preview the Slurm commands and resource requests, then submit the workflow after reviewing them:

```bash
bash runs/run.sh submit-training --dry-run
bash runs/run.sh submit-training
```

Submission schedules the following stages with dependencies, so each starts after the preceding jobs succeed:

1. **Prepare inputs:** repeat input preparation and validation on the scheduled jobs.
2. **Pretrain models:** train the merged and lineage configurations for 10 epochs by default, saving checkpoints every half epoch.
3. **Evaluate checkpoints:** fine-tune and evaluate each candidate on classification and age-prediction tasks.
4. **Select models:** choose one checkpoint per cohort using the downstream evaluation metrics.
5. **Prepare analysis baselines:** prepare perturbation task manifests, fit and save one final age model per lineage, reuse the selected merged classifier, and verify all five baselines and GPU no-op controls. Perturbation tasks reuse these models without fitting another head.

By default, each downstream evaluation fine-tunes a fresh task head and backbone for one epoch at learning rate 0.000804, weight decay 0.01, warmup 100 steps and seed 73. Classification includes broad cell type, cluster and location; age regression includes five-fold donor-grouped cross-validation within the training donors and evaluation on the fixed held-out donors.

Selection uses the mean of cluster macro-F1, location macro-F1 and held-out age Pearson correlation, with lower pretraining validation loss breaking ties. A classification task with only one training class is marked as not estimable. Broad-cell-type metrics are reported but are not part of selection. Because held-out metrics select the model, they describe selection-set performance; an independent final test requires an additional donor cohort that was not used for selection.

## 3. Review training and validation results

The submission command returns after scheduling jobs; it does not wait for training to finish. Wait for training, evaluation and baseline verification to complete successfully before submitting perturbation screens. Inspect job logs under `logs/` and the submission record at `diagnostics/submission_training.json`.

```bash
bash runs/run.sh report
```

Open `outputs/web_summary/index.html` to review training loss curves, downstream metrics and model-selection results. Rankings and chosen checkpoints are under `outputs/model_selection/`. For each cohort, `outputs/model_registry/<group>/` contains `binding.json`, `noop_controls.json` and `baseline_verified.json`, which bind the saved model and inputs to the unperturbed predictions used for analysis. Perturbation submission checks that these records still match the model, inputs and baseline files.

### Individual stages and recovery

Run an individual stage with `bash runs/run.sh STAGE` on a node with the required resources. Recovery stages include `train --task-id N`, `downstream --task-id N`, `select --group GROUP`, `prepare-screens` and `fit-baseline --group GROUP`. Valid group names are `delight_fm`, `immune`, `epithelium_colon`, `epithelium_ileum` and `stromal`.

Training task IDs are listed in `conditions.tsv`; the downstream task ID is the training ID × number of checkpoint epochs + the zero-based checkpoint index. `train`, `downstream`, `perturb` and `genes` use global task IDs rather than `--group`. Before retrying a failed submission, inspect the stored scheduler commands and logs; submission records prevent accidentally launching the full workflow twice.

## 4. Run gene and regulon perturbation screens

After training and baseline verification complete successfully:

```bash
bash runs/run.sh submit-perturbations --dry-run
bash runs/run.sh submit-perturbations
```

The merged model screens curated marker genes and CollecTRI regulons for changes in cell-type classification. The four lineage age models screen individual genes and SCENIC+ regulons for changes in predicted log2-transformed age in post-conception weeks (PCW). Within each lineage, all screens share one saved age model and the same verified baseline predictions.

The submission schedules the screens, merges complete verified shards and generates reports. Default all-gene arrays process 64 genes per shard, saving progress every 16 genes. Targets with no model token coverage are recorded explicitly; token coverage differs from RNA detection percentage.

## 5. Run proportional BHLHE40 perturbations

After the ileum saved age model and baseline pass verification:

```bash
bash runs/run.sh submit-bhlhe40 --dry-run
bash runs/run.sh submit-bhlhe40
```

The ileum workflow uses bulk-expression-informed proportional deletion and overexpression at strengths 0.25, 0.5 and 1, plus TF-only and targets-only controls. Deletion applies the inverse of the bulk overexpression response; it does not represent a measured knockout signature. Both directions share the saved ileum age model and baseline. Reports include donor prediction-change dotplots and donor mean predicted-age lines against baseline. The configured target table is ileum-specific; applying it to another lineage requires an appropriate target source before planning.

## 6. Cluster regulon responses and generate reports

After the regulon results have finished merging, cluster their effects on predicted age across donors and refresh the reports:

```bash
bash runs/run.sh cluster
bash runs/run.sh report
# Or one lineage:
bash runs/run.sh cluster --group epithelium_ileum
```

Clustering uses balanced magnitude-aware Ward clustering with 12 groups by default. A shared scaling transformation preserves the direction and relative magnitude of regulon effects across donors. Outputs include representative regulons and checks of how stable the groups are when individual donors are left out.

Reports and clustering use completed outputs from the active run. Reports can be regenerated while outputs accumulate; absent analyses are omitted. Clustering requires a complete regulon-by-donor matrix of prediction changes. `REPORT_PYTHON_BIN` selects the reporting environment for both commands.

| Output | Location under this run |
|---|---|
| Model-selection rankings, chosen checkpoints | `outputs/model_selection/` |
| Saved-model/baseline binding and GPU controls | `outputs/model_registry/<group>/` |
| Canonical saved lineage age model and all-gene baseline | `outputs/perturb_age_all_genes/<group>/age_model/` and adjacent files |
| All-gene summaries/per-donor deltas | `outputs/perturb_age_all_genes/<group>/` |
| Merged marker/CollecTRI outputs | `outputs/delight_fm/markers/`, `outputs/delight_fm/regulons/` |
| SCENIC+ outputs | `outputs/<group>/regulons/` |
| BHLHE40 proportional outputs | `outputs/epithelium_ileum/bhlhe40/{delete,overexpress}/` |
| Main report index | `outputs/web_summary/index.html` |
| Training/downstream plots | `outputs/web_summary/training/figures/` |
| Individual model summaries and raw figures | `outputs/<group>/web_summary/index.html` and `figures/` |
| k=12 cluster report, assignments, features, figures | `outputs/<group>/regulon_clusters_k12/` |

Individual summaries embed PNG images; PDF/PNG figures and CSV data are stored beside each report. Merged reports emphasize classifier changes, class contributions and UMAP embedding L2 shifts. Raw log2(PCW) donor plots belong to lineage age models. Source details are in [the source README](../src/README.md).

## Validation and scope

CPU tests cover signed float32 edits, unchanged-input controls, mask alignment, baseline mismatch rejection, clustering scale, launcher portability and command compatibility. Passing these tests does not replace the GPU baseline and no-op checks required by each analysis run. The workflow creates new checkpoints; published checkpoints and model-specific inference notebooks are available in the [Hugging Face model family](https://huggingface.co/chloehlee/).

The age report includes an early-life mean absolute effect ranking. For an independent final evaluation, reserve an additional donor cohort beyond those used for checkpoint selection.

## Published checkpoints and notebooks

To use existing checkpoints directly, follow the [model task guide](../docs/MODEL_TASKS.md). The merged checkpoint supports saved-head cell-type and cluster prediction plus marker/CollecTRI cell-type perturbations. Lineage foundation notebooks cover saved classification heads and task fine-tuning; the matching `-age` packages exclusively provide log2(PCW) prediction and gene/SCENIC+ perturbations. The Slurm workflow above performs a fresh sweep and final fitting rather than downloading those fitted checkpoints into its registry.
