# Run DELIGHT-FM v1

Run these commands from the repository root. Install the pinned modeling environment and set `PYTHON_BIN` to its Python executable. Install the [published input dataset](https://huggingface.co/datasets/chloehlee/delight_fm_v1-data) using the [dataset instructions](../docs/dataset/README.md). Configuration and donor assignments are relative to this run directory.

The launchers do not load environment modules. Prepare CUDA and any site-specific modules before running them, and set `REPORT_PYTHON_BIN` for the separate reporting environment. Optional RNA rebuilding uses `RSCRIPT_BIN` (default: `Rscript` on PATH). See the [environment setup](../README.md#set-up-the-environments). Scheduled jobs use the launchers from the current Git checkout, with data and outputs in the selected run directory.

The `submit-*` commands require Slurm. Set `slurm.gpu_partition` and `slurm.cpu_partition` in `config/settings.json` for your cluster; `null` uses the scheduler default without adding a partition argument. Review all resource settings before submission. To use another scheduler or run directly, invoke individual stages with the appropriate task IDs and allocate the required CPU/GPU resources yourself.

## Cohorts

| Model | Train cells | Test cells | Train donors | Test donors |
|---|---:|---:|---:|---:|
| Merged DELIGHT-FM | 531,101 | 140,257 | 128 | 21 |
| Immune | 175,031 | 36,586 | 126 | 21 |
| Colon epithelium | 85,035 | 25,155 | 79 | 13 |
| Ileum epithelium | 67,849 | 14,564 | 71 | 13 |
| Stromal | 206,814 | 60,323 | 119 | 21 |

These counts were checked against actual tokenized metadata, with no donor overlap. Colon preserves the latest two <1-year assignments and training teenager donor. Ileum excludes Adult, 80+; colon and ileum each require at least 50 training cells in every retained age group. Other cohorts retain their original splits. Each cohort contains its available subset of donors; the donor sets are not identical across lineages. Per-age cell/donor counts are exported to `diagnostics/*_age_counts.csv` by validation.

## 1. Configure and freeze

Review the hyperparameter grids, checkpoint epochs, marker/TF panels and Slurm resources before planning. Defaults are 6 merged configurations and 4 per lineage (22 training jobs), with 20 half-epoch evaluations from 0.5 to 10 epochs (440 downstream jobs). CollecTRI defaults to a configured 13-TF panel; set `perturbation.collectri_tfs` to `"all"` before planning to use the full bundled human table. Lineage regulons use the bundled SCENIC+ tables.

```bash
bash runs/run.sh plan
bash runs/run.sh prepare
bash runs/run.sh validate
bash runs/run.sh submit-training --dry-run
```

`plan` is immutable and can be repeated unchanged. Create a new plan after installation; computational changes require a fresh run/plan. Slurm resources and clustering resolution can change without invalidating fitted models. Copy the configuration, references, donors and local inputs to another run directory and set `V12_RUN_DIR` when starting a different analysis; leave generated results/diagnostics behind. `rebuild-inputs --group GROUP` is an optional fresh RNA export/tokenization route that requires empty exported/tokenized destinations. It was syntax checked, while the local datasets were validated against all five fixed donor splits.

## 2. Train and evaluate

```bash
bash runs/run.sh submit-training
```

Dependency graph: prepare inputs -> full pretraining sweep -> all checkpoint downstream evaluations -> select one checkpoint per cohort -> prepare perturbation task manifests -> fit/save each lineage age model and verify all five baselines. The merged classifier reuses its selected saved pretrained model. No perturbation task fits another head.

Each downstream evaluation fine-tunes a fresh task head and backbone for one epoch at LR 0.000804, weight decay 0.01, warmup 100 steps, seed 73. Classification includes broad cell type, cluster and location; age regression includes five-fold donor-grouped training CV and evaluation on the fixed held-out donors. Selection preserves the final workflow: mean of cluster macro-F1, location macro-F1 and held-out age Pearson, with lower pretraining validation loss breaking ties; a single-class classification task is explicitly not estimable. Broad-cell-type metrics are plotted but not part of selection. Because held-out metrics select the model, they are selection-set performance rather than an independent final test.

Individual stages are available for recovery: `train --task-id N`, `downstream --task-id N`, `select --group GROUP`, `prepare-screens`, `fit-baseline --group GROUP`. Training IDs are rows in `conditions.tsv`; downstream ID is training ID × number of checkpoint epochs + zero-based checkpoint index. `train`, `downstream`, `perturb`, and `genes` use global task IDs, not `--group`. Before retrying a failed submission, inspect the stored scheduler commands and logs; submission records prevent accidentally launching a duplicate full DAG.

## 3. Shared-model perturbation screens

After training and baseline verification complete successfully:

```bash
bash runs/run.sh submit-perturbations --dry-run
bash runs/run.sh submit-perturbations
```

This runs merged curated markers and CollecTRI classifier perturbations, plus all-gene and SCENIC+ age screens for all four lineages; merges require complete verified shards. The report runs after merging. Default all-gene arrays are sharded into 64 genes, with commits every 16 genes. Zero-coverage targets are recorded explicitly. Model token coverage is distinct from RNA detection percentage.

## 4. Proportional BHLHE40

After the ileum saved age model and baseline pass verification:

```bash
bash runs/run.sh submit-bhlhe40 --dry-run
bash runs/run.sh submit-bhlhe40
```

Only bulk-informed proportional deletion/overexpression is included: strengths 0.25, 0.5 and 1, plus TF-only and targets-only controls. Both directions share the ileum saved age model and baseline. Reports include donor prediction-change dotplots and donor mean predicted-age lines against baseline. The configured target table is ileum-specific; applying it to another lineage needs a justified target source before planning.

## 5. Reports and balanced magnitude-aware clustering

```bash
bash runs/run.sh report
bash runs/run.sh cluster
# Or one lineage:
bash runs/run.sh cluster --group epithelium_ileum
```

The reports and clustering use completed outputs from the configured run. Reports can be regenerated while outputs accumulate; absent analyses are omitted. Clustering requires complete regulon-by-donor values.

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

CPU tests cover signed float32 no-ops/ties, mask alignment, baseline mismatch rejection, clustering scale and command compatibility. Model engines import in the pinned environment. Training/classifier/age/BHLHE40 reports rendered from completed historical results in an isolated diagnostic fixture; the new k=12 renderer reproduces the published ileum feature matrix/common scale and clusters (ARI=1). The full training commands create new checkpoints; published checkpoints and model-specific inference notebooks are available in the [Hugging Face model family](https://huggingface.co/chloehlee/). GPU fitting and mandatory baseline/no-op prediction checks remain to run in the submitted workflow.

Useful next additions, kept outside this minimal first collection: expression-vs-deletion proportion/dotplot reports using the RNA RDS, fetal/adult/early-life composite programs and Seurat expression dotplots, and dedicated fetal-to-early-life regulon contrasts. The present age report retains an early-life mean absolute effect ranking. For publication-level independent generalization, reserve an additional untouched donor cohort beyond the donors used for checkpoint selection.

## Published checkpoints and notebooks

To use existing checkpoints directly, follow the [model task guide](../docs/MODEL_TASKS.md). The merged checkpoint supports saved-head cell-type and cluster prediction plus marker/CollecTRI cell-type perturbations. Lineage foundation notebooks cover saved classification heads and task fine-tuning; the matching `-age` packages exclusively provide log2(PCW) prediction and gene/SCENIC+ perturbations. The Slurm workflow above performs a fresh sweep and final fitting rather than downloading those fitted checkpoints into its registry.
