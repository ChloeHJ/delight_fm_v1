# DELIGHT-FM v1 source

This guide maps the source modules and describes implementation details and developer checks. For setup and data installation, see the [main README](../README.md); for workflow commands, configuration and outputs, see the [run guide](../runs/README.md). Published model capabilities and notebooks are listed in the [model task guide](../docs/MODEL_TASKS.md).

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
| `cluster_regulons.py` | Regulon clustering features, representative selection and donor plots |
| `plots.py`, `figure_style.py`, `donor_plotting.py`, `report_assets.py` | Training/downstream/classifier/age/BHLHE40 plots, PDF/PNG/CSV, embedded HTML images |

## Inference and editing details

Inference uses float32, disables autocast and TF32, and strictly reloads saved weights. Hashes bind the model, input metadata, expression statistics, reference predictions and baseline controls. Changes to weights, cell ordering or baseline predictions fail verification. GPU no-op controls check that zero-strength, zero-fold-change and absent-target edits leave model inputs and predictions unchanged before screens proceed. Shards record completion so screens can resume without refitting the age model.

The BHLHE40 proportional editor reconstructs approximate expression from standardised inputs, applies the signed bulk fold change, then standardises and clips the edited values in float32. Changes too small to affect float32 values preserve the original token order and aligned masks.

The local padding adapter handles right padding and special tokens, while Hugging Face’s `DataCollatorForLanguageModeling` implements masking. An external editable Geneformer installation is not required.

## Clustering implementation

`cluster_regulons.py` defines a common scale as the median regulon root mean square (RMS) across donors. It transforms each entry of the signed prediction-change matrix with `asinh(delta/common_scale)`, then applies Euclidean Ward linkage.

There is no row centring or per-regulon variance normalisation. PCA is used for display only. Representatives are cluster medoids plus a second central regulon from a different TF where available. For commands, defaults and report outputs, see the [clustering instructions](../runs/README.md#6-cluster-regulon-responses-and-generate-reports).

## Developer checks

After completing the [environment setup](../README.md#set-up-the-environments), run CPU tests from the repository root. The padding test uses PyTorch and Transformers from the modeling environment:

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
