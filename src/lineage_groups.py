"""Summarize fixed-model cell effects within original atlas Lineage groups."""
import numpy as np
import pandas as pd

SCORE_DEFINITION = "annotated_class_probability_change_within_original_lineage"


def load_original_lineages(path, cell_ids):
    mapping = pd.read_csv(path, usecols=["cell_id", "OriginalLineage"])
    if mapping.cell_id.duplicated().any() or mapping.OriginalLineage.isna().any():
        raise ValueError("Original Lineage mapping must contain unique cells and complete annotations")
    labels = mapping.set_index("cell_id").OriginalLineage.reindex(list(map(str, cell_ids)))
    if labels.isna().any():
        raise ValueError(f"Original Lineage missing for {int(labels.isna().sum())} perturbation cells")
    return labels.to_numpy(dtype=str)


def grouped_cell_effects(perturbation, lineages, present, delta, l2_shift, perturb_type="delete"):
    lineages = np.asarray(lineages)
    present = np.asarray(present, dtype=bool)
    delta, l2_shift = np.asarray(delta, dtype=float), np.asarray(l2_shift, dtype=float)
    if not (len(lineages) == len(present) == len(delta) == len(l2_shift)):
        raise ValueError("Lineage, presence and effect arrays must describe the same cells")
    direction = -1. if perturb_type == "delete" else 1.
    rows = []
    for label in sorted(set(lineages)):
        group = lineages == label
        expressed = group & present
        row = dict(input_gene=perturbation["input_gene"],
                   perturbation_label=perturbation.get("perturbation_label", perturbation["input_gene"]),
                   perturbation_kind=perturbation.get("perturbation_kind", "single_gene"),
                   lineage=label, grouping_column="OriginalLineage", score_definition=SCORE_DEFINITION,
                   perturb_type=perturb_type, n_cells=int(group.sum()),
                   n_cells_with_gene=int(expressed.sum()), fraction_cells_with_gene=float(present[group].mean()))
        for mask, prefix in [(group, ""), (expressed, "present_")]:
            row[prefix + "delta_cell_class_prob_mean"] = float(np.mean(delta[mask])) if mask.any() else np.nan
            row[prefix + "lineage_contribution_score_mean"] = direction * row[prefix + "delta_cell_class_prob_mean"]
            row[prefix + "embedding_l2_shift_mean"] = float(np.mean(l2_shift[mask])) if mask.any() else np.nan
        rows.append(row)
    return rows


def correlation_linkages(matrix):
    """Average linkage; use Euclidean distances when constant profiles prevent correlation."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist
    values = np.nan_to_num(matrix.to_numpy(dtype=float), nan=0., posinf=0., neginf=0.)
    result = []
    for data in [values, values.T]:
        if len(data) < 2:
            result.append(None)
            continue
        distances = pdist(data, metric="correlation")
        if not np.isfinite(distances).all():
            distances = pdist(data, metric="euclidean")
        result.append(linkage(distances, method="average")
                      if np.isfinite(distances).all() and np.any(distances > 0) else None)
    return tuple(result)


def cluster_rows(matrix):
    from scipy.cluster.hierarchy import leaves_list
    rows, _ = correlation_linkages(matrix)
    return matrix.iloc[leaves_list(rows)] if rows is not None else matrix
