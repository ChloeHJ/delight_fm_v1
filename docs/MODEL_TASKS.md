# Model tasks and notebooks

Five foundation checkpoints and four fitted age checkpoints are public in [chloehlee’s DELIGHT-FM family](https://huggingface.co/chloehlee/). See the [published model table](../README.md#published-models) and [revision snapshot](PUBLISHED_REVISIONS.md).

| Task | Foundation merged | Foundation lineage | Fitted lineage `-age` |
|---|---|---|---|
| Broad-cell classification with saved pretraining labels | Saved 17-class head | Saved compartment-specific head | Use foundation companion |
| Cluster classification with saved pretraining labels | Saved 62-class head | Saved compartment-specific head | Use foundation companion |
| New cell-type or cluster labels | Fine-tune and save a task head | Fine-tune and save a task head | Use foundation companion for new tasks |
| Location classification | Fine-tune location head | Immune/stromal only; fine-tune location head | Use foundation companion |
| Predicted developmental age, log2(PCW) | No fitted age companion in this release | Use age companion | Saved age head and fine-tuned backbone |
| Individual gene and SCENIC+ age deletion | Use lineage age models | Documented in age companion only | Supported directly |
| Marker gene and CollecTRI broad-cell deletion | Supported using saved broad-cell head | Use merged model for this workflow | Use merged model |

Foundation notebook order is:

1. `01_saved_head_classification.ipynb` — inference using that repository's saved broad-cell/cluster heads.
2. `02_verified_perturbations.ipynb` — merged DELIGHT-FM only, using its own saved broad-cell head for marker/CollecTRI deletion.
3. `03_finetune.ipynb` — fit a new classifier starting from that repository's foundation backbone.

Lineage foundation packages contain `01_saved_head_classification.ipynb` and `02_finetune.ipynb`; their age perturbations belong exclusively to the matching `-age` package. Each age package contains `01_predict_age.ipynb` and `02_gene_scenic_age_perturbations.ipynb`, both loading its own saved age model. No workflow hosted in one model repository loads a different model repository as its primary model.

Foundation auxiliary PCW scores are not calibrated log2(PCW). Pretraining broad-cell/cluster heads use the released label maps and differ from separately fine-tuned downstream evaluation heads. Those task-specific evaluation heads are not included in the foundation weights. Colon and ileum each have one tissue location, so location classification is not meaningful within those cohorts.

SCENIC+ and CollecTRI notebooks use joint token deletion, optionally including the TF. Edge signs identify programs or regulatory evidence; they are not treated as measured bulk fold changes. Proportional BHLHE40 signed editing remains available in the GitHub proportional workflow and package signed editor. Every screen reuses one saved model, matching released statistics and one verified baseline. Examples run empty/absent deletion controls before edits and export model/baseline/program bindings and token coverage.

New input AnnData must use compatible natural-log1p library-normalized expression and tissue-matched preprocessing. Exact reproduction on original cells should reuse the released tokenized rows. Notebook code and CPU helper semantics were checked; GPU fitting, inference and calibration on user data must be evaluated in the compatible CUDA environment.

## Open published notebooks

| Model | Notebook directory | Regulatory source for perturbations |
|---|---|---|
| Merged atlas foundation | [Classification and fine-tuning](https://huggingface.co/chloehlee/delight-fm/tree/main/notebooks) | CollecTRI, broad-cell probability effects |
| Ileum epithelium foundation | [Classification and fine-tuning](https://huggingface.co/chloehlee/delight-fm-epi-ileum/tree/main/notebooks) | Age perturbations are in the fitted age package |
| Ileum epithelium age | [Prediction and perturbations](https://huggingface.co/chloehlee/delight-fm-epi-ileum-age/tree/main/notebooks) | Compartment SCENIC+ reconstructed regulons, log2(PCW) effects |
| Colon epithelium foundation | [Classification and fine-tuning](https://huggingface.co/chloehlee/delight-fm-epi-colon/tree/main/notebooks) | Age perturbations are in the fitted age package |
| Colon epithelium age | [Prediction and perturbations](https://huggingface.co/chloehlee/delight-fm-epi-colon-age/tree/main/notebooks) | Compartment SCENIC+ reconstructed regulons, log2(PCW) effects |
| Immune foundation | [Classification and fine-tuning](https://huggingface.co/chloehlee/delight-fm-immune/tree/main/notebooks) | Age perturbations are in the fitted age package |
| Immune age | [Prediction and perturbations](https://huggingface.co/chloehlee/delight-fm-immune-age/tree/main/notebooks) | Compartment SCENIC+ reconstructed regulons, log2(PCW) effects |
| Stromal foundation | [Classification and fine-tuning](https://huggingface.co/chloehlee/delight-fm-stromal/tree/main/notebooks) | Age perturbations are in the fitted age package |
| Stromal age | [Prediction and perturbations](https://huggingface.co/chloehlee/delight-fm-stromal-age/tree/main/notebooks) | Compartment SCENIC+ reconstructed regulons, log2(PCW) effects |
