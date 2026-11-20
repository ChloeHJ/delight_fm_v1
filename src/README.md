# DELIGHT-FM v1 source

This directory contains the code for reproducing DELIGHT-FM input preparation, pretraining, task-specific fine-tuning and downstream analyses. The optional preparation route starts from Seurat RDS files and exports expression data and metadata for tokenisation. The standard workflow uses the supplied tokenised datasets, evaluates candidate models and applies gene and regulon perturbations using verified saved models.

Start with the main README’s [environment setup](../README.md#set-up-the-environments) and [data installation](../README.md#install-data), then follow the [run guide](../runs/README.md) for commands and output locations. The dataset helper pins a Hugging Face commit revision and verifies input files against the SHA-256 manifest. Published model capabilities and usage notebooks are listed in the [model task guide](../docs/MODEL_TASKS.md).

## Source modules

| Module | Responsibility |
|---|---|
| `pipeline.py`, `common.py`, `paths.py` | CLI, immutable run plan, input/source hashes, Slurm dependencies |
| `notebook_workflows.py` | Portable target resolution, no-op controls and donor/classifier summaries included in published notebook packages |
| `prepare_inputs.py`, `prepare_inputs.R`, `build_atlas_hf_dataset.py` | Validate donor splits and input metadata; optionally export Seurat RDS inputs and tokenise cells using fixed donors |
| `training.py`, `train_atlas_mamba.py`, `geneformer_mamba_model.py` | Model architecture, pretraining sweep, half-epoch checkpoints and selection of one checkpoint per cohort |
| `compare_mamba_on_downstream.py`, `token_padding.py` | Downstream fine-tuning/evaluation; local padding adapter with unchanged Hugging Face MLM masking |
| `screens.py`, `baseline_contract.py`, `controls.py` | One model/reference per cohort; model binding, reload checks, mandatory GPU no-op controls |
| `perturb_lineage.py`, `lineage_groups.py` | Merged cell-type marker/CollecTRI perturbations and classifier/embedding effects |
| `perturb_age.py`, `donor_scores.py` | SCENIC+ lineage age perturbations and donor aggregation |
| `all_gene_age.py`, `all_gene_age_runtime.py`, `all_gene_age_stats.py` | Fit lineage age head once; resumable token-indexed all-gene screens, merge/rank |
| `bhlhe40.py`, `signed_tf_targets.py` | Bulk-expression-informed proportional BHLHE40 deletion and overexpression in the ileum age model |
| `cluster_regulons.py` | Group regulons by donor-level predicted-age responses using balanced magnitude-aware Ward clustering; 12 groups by default |
| `plots.py`, `figure_style.py`, `donor_plotting.py`, `report_assets.py` | Training/downstream/classifier/age/BHLHE40 plots, PDF/PNG/CSV, embedded HTML images |

## From inputs to analysis

The pipeline runs through the following stages:

1. **Prepare and validate inputs:** check fixed donor assignments, train/test separation, cell counts, age labels, sequence lengths and age-group coverage.
2. **Pretrain and evaluate candidates:** train the configured merged and lineage models, save checkpoints every half epoch by default, and fine-tune candidates for classification and developmental-age prediction. Age evaluation uses donor-grouped cross-validation within the training donors and the fixed held-out donors.
3. **Select and verify models:** select a checkpoint per cohort, fit one final age model per lineage, and save and verify the models and baseline predictions that the screens will share.
4. **Run perturbations:** measure changes in merged-model cell-type classification or lineage-model predicted age after gene or regulon edits.
5. **Summarise results:** aggregate donor responses, cluster regulon response patterns and generate figures, tables and HTML reports.

Defaults and selection metrics are described in the [training and evaluation instructions](../runs/README.md#2-train-and-evaluate-candidate-models). Held-out metrics are used for checkpoint selection, so they describe selection-set performance; an independent final test requires an additional donor cohort.

## Saved models and baseline verification

Merged-model perturbations use the selected saved model’s cell-type classifier. Developmental-age analyses use the four lineage age models, which predict log2-transformed age in post-conception weeks (PCW). Within each lineage, all-gene and SCENIC+ screens reuse one saved age model and the same held-out-cell baseline predictions. The BHLHE40 analysis shares the ileum age model and baseline. Auxiliary age outputs from foundation models are not calibrated log2(PCW) predictions.

Inference uses float32, disables autocast and TF32, and strictly reloads saved weights. Hashes bind the model, input metadata, expression statistics, reference predictions and baseline controls. Changes to weights, cell ordering or baseline predictions fail verification. GPU no-op controls check that zero-strength, zero-fold-change and absent-target edits leave model inputs and predictions unchanged before screens proceed. Shards record completion so screens can resume without refitting the age model.

The BHLHE40 proportional editor reconstructs approximate expression from standardised inputs, applies the signed bulk fold change, then standardises and clips the edited values in float32. Changes too small to affect float32 values preserve the original token order and aligned masks. Deletion applies the inverse bulk overexpression response; it does not represent a measured knockout signature.

## Clustering

When grouping regulons by their perturbation effects on predicted age across donors, `cluster_regulons.py` constructs a signed regulon-by-donor matrix of mean prediction changes. It defines a common scale as the median regulon root mean square (RMS) across donors, transforms each entry with `asinh(delta/common_scale)`, then applies Euclidean Ward linkage and cuts the tree into 12 groups by default.

The shared scaling preserves the direction and relative magnitude of effects, without row centring or per-regulon variance normalisation. PCA is used for display only. Representatives are cluster medoids plus a second central regulon from a different TF where available. Reports include donor plots, leave-one-donor-out agreement and a scan of alternative cluster counts. Run clustering after the regulon results have finished merging, as described in the [clustering instructions](../runs/README.md#6-cluster-regulon-responses-and-generate-reports).

## Environments and checks

Follow the main README’s [environment setup](../README.md#set-up-the-environments). [Modeling requirements](requirements-model.txt) record the Python 3.10/CUDA 11.8 stack; compiled Mamba extensions must match PyTorch. [Reporting requirements](requirements-report.txt) describe the separate plotting and clustering environment.

The launcher uses `PYTHON_BIN` for modeling and `REPORT_PYTHON_BIN` for reports and clustering; each defaults to `python3` on PATH. It clears inherited `PYTHONPATH` for modeling stages and does not load environment modules automatically. Optional RNA rebuilding uses `RSCRIPT_BIN`, defaulting to `Rscript` on PATH. Select the active run with `DELIGHT_FM_RUN_DIR`; the [run guide](../runs/README.md) explains run selection, Slurm configuration and recovery commands.

The local padding adapter handles right padding and special tokens, while Hugging Face’s `DataCollatorForLanguageModeling` implements masking. An external editable Geneformer installation is not required.

Run CPU tests from the repository root using the environment appropriate to each test. The padding test uses PyTorch and Transformers from the modeling environment:

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 "$PYTHON_BIN" -m unittest discover \
  -s src/tests -p test_token_padding.py -v
```

Run the numerical, notebook-helper and portability tests in the reporting environment:

```bash
for test_file in test_final_workflow.py test_notebook_workflows.py test_portability.py; do
  env -u PYTHONPATH PYTHONNOUSERSITE=1 "$REPORT_PYTHON_BIN" -m unittest discover \
    -s src/tests -p "$test_file" -v || break
done
```

These tests cover signed float32 edits, mask alignment, baseline mismatch rejection, donor summaries, clustering and portable launchers. Each analysis run must also pass its GPU baseline and no-op prediction checks before perturbation screens proceed.
