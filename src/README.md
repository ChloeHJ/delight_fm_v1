# DELIGHT-FM v1 source

See [run instructions](../runs/README.md). Input datasets are available from [Hugging Face](https://huggingface.co/datasets/chloehlee/delight_fm_v1-data) and versioned by SHA-256 manifest. Published checkpoint capabilities and notebook links are listed in the [model task guide](../docs/MODEL_TASKS.md).

| Module | Responsibility |
|---|---|
| `pipeline.py`, `common.py`, `paths.py` | CLI, immutable run plan, input/source hashes, Slurm dependencies |
| `notebook_workflows.py` | Portable target resolution, no-op controls and donor/classifier summaries included in published notebook packages |
| `prepare_inputs.py`, `prepare_inputs.R`, `build_atlas_hf_dataset.py` | Validate local inputs; optional RNA rebuild using fixed donors |
| `training.py`, `train_atlas_mamba.py`, `geneformer_mamba_model.py` | Full configuration and half-epoch sweeps; pretrained classifier backbone |
| `compare_mamba_on_downstream.py`, `token_padding.py` | Downstream fine-tuning/evaluation; local padding adapter with unchanged Hugging Face MLM masking |
| `screens.py`, `baseline_contract.py`, `controls.py` | One model/reference per cohort; model binding, reload checks, mandatory GPU no-op controls |
| `perturb_lineage.py`, `lineage_groups.py` | Merged cell-type marker/CollecTRI perturbations and classifier/embedding effects |
| `perturb_age.py`, `donor_scores.py` | SCENIC+ lineage age perturbations and donor aggregation |
| `all_gene_age.py`, `all_gene_age_runtime.py`, `all_gene_age_stats.py` | Fit lineage age head once; resumable token-indexed all-gene screens, merge/rank |
| `bhlhe40.py`, `signed_tf_targets.py` | Bulk-informed proportional BHLHE40 deletion/overexpression only |
| `cluster_regulons.py` | Balanced magnitude-aware Ward, k=12, representative regulons and donor plots |
| `plots.py`, `figure_style.py`, `donor_plotting.py`, `report_assets.py` | Training/downstream/classifier/age/BHLHE40 plots, PDF/PNG/CSV, embedded HTML images |

## Model and baseline contract

Merged perturbations use the selected saved model's cell-type classifier. Its auxiliary transformed age output is not used for log2(PCW) interpretation. Each lineage fits one final regression model to log2(PCW), saves it once, and shares its held-out-cell baseline between all genes, SCENIC+ regulons and BHLHE40. Donor plots use those lineage regression outputs.

Inference explicitly uses float32, disables autocast and TF32, and strictly reloads saved weights. Model, input metadata, expression statistics, reference predictions and baseline controls are hashed. Changed weights/cell ordering/predictions fail verification. Zero-strength, zero-fold-change and absent-target perturbations must pass GPU checks before screens proceed. Shards carry completion manifests and can resume without refitting the head.

Proportional editing reconstructs approximate expression from standardized inputs, applies the signed bulk fold change, restandardizes/clips, and edits in float32. Ineffective float32 changes preserve the exact original token order and aligned masks. Deletion imposes the inverse bulk overexpression response; it is not a measured knockout signature. Binary/promoter-only/abundance-multiplier experiments are not retained.

## Clustering

Construct the signed regulon-by-donor matrix of mean prediction changes. Define a common scale as the median regulon RMS across donors; transform every entry by `asinh(delta/common_scale)`. Apply Euclidean Ward linkage and cut into 12 groups. There is no row centering or per-regulon variance normalization. Clustering preserves sign, offset and amplitude; PCA is for display only. Representatives are cluster medoids plus a second central regulon from a different TF where available. Leave-one-donor-out agreement and a resolution scan accompany the plots. This reproduces the final ileum 303-regulon/13-donor clustering exactly.

## Environments and checks

`requirements-model.txt` records the pinned Python 3.10/CUDA 11.8 modelling stack; compiled Mamba extensions must match PyTorch. `requirements-report.txt` describes the separate plotting environment. `run.sh` clears inherited PYTHONPATH for the model environment and uses `PYTHON_BIN` for modeling and `REPORT_PYTHON_BIN` for reports; each defaults to `python3` on PATH. No environment modules are loaded automatically. Optional RNA rebuilding uses `RSCRIPT_BIN`, defaulting to `Rscript` on PATH. See the [environment setup](../README.md#set-up-the-environments). No editable external Geneformer installation is needed: the local input-only adapter follows Geneformer's [right padding and special-token handling](https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/pretrainer.py), while `DataCollatorForLanguageModeling` still implements masking.

From the Git repository root, with `PYTHON_BIN` set to the modeling environment:

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 "$PYTHON_BIN" -m unittest discover -s src/tests -v
```
