# DELIGHT-FM v1 source

This guide maps the source modules. For setup and data installation, see the [main README](../README.md); for workflow commands, configuration and outputs, see the [run guide](../runs/README.md). Published model capabilities and notebooks are listed in the [model task guide](../docs/MODEL_TASKS.md).

## Source modules

| Module | Responsibility |
|---|---|
| `pipeline.py`, `common.py`, `paths.py` | CLI, immutable run plan, input/source hashes, Slurm dependencies |
| `notebook_workflows.py` | Donor/classifier summaries included in published notebook packages |
| `prepare_inputs.py`, `prepare_inputs.R`, `build_atlas_hf_dataset.py` | Validate donor splits and input metadata; optionally export Seurat RDS inputs and tokenise cells using fixed donors |
| `training.py`, `train_atlas_mamba.py`, `geneformer_mamba_model.py` | Model architecture, pretraining sweep, half-epoch checkpoints and selection of one checkpoint per cohort |
| `compare_mamba_on_downstream.py`, `token_padding.py` | Downstream fine-tuning/evaluation; local padding adapter with unchanged Hugging Face MLM masking |
| `screens.py`, `baseline_contract.py`, `controls.py` | One model/reference per cohort; model binding, reload checks, mandatory GPU no-op controls |
| `perturb_lineage.py`, `lineage_groups.py` | Cell-type marker/CollecTRI perturbations and classifier/embedding effects |
| `perturb_age.py`, `donor_scores.py` | SCENIC+ lineage age perturbations and donor aggregation |
| `all_gene_age.py`, `all_gene_age_runtime.py`, `all_gene_age_stats.py` | Fit lineage age head once; resumable token-indexed all-gene screens, merge/rank |
| `bhlhe40.py`, `signed_tf_targets.py` | Bulk-expression-informed proportional BHLHE40 deletion and overexpression in the ileum age model |
| `cluster_regulons.py` | Regulon clustering features, representative selection and donor plots |
| `plots.py`, `figure_style.py`, `donor_plotting.py`, `report_assets.py` | Training/downstream/classifier/age/BHLHE40 plots, PDF/PNG/CSV, embedded HTML images |

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
