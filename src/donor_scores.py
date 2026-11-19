"""Age perturbation summaries with one row per biological donor and experiment."""
import numpy as np
import pandas as pd


def _median(values):
    return float(values.median()) if values.notna().any() else np.nan


def donor_summary(per_cell: pd.DataFrame, donor_column: str) -> pd.DataFrame:
    group_cols = ["perturbation_label", "perturbation_kind", donor_column]
    rows = []
    for keys, group in per_cell.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        # Libraries and other metadata must never split a biological donor.
        for column in ("age_contribution_program", "donor_id", "SampleID", "PCW", "log2_PCW", "AgeGroup", "pcw_bin", "Location"):
            if column not in group.columns or column in row:
                continue
            values = group[column].dropna().unique()
            if len(values) > 1 and column in {"age_contribution_program", "donor_id", "PCW", "log2_PCW", "AgeGroup"}:
                raise ValueError(f"Conflicting {column} within donor {row[donor_column]}: {values}")
            row[column] = values[0] if len(values) == 1 else ",".join(sorted(map(str, values))) if len(values) else np.nan
        row.update(
            {
                "n_cells": int(len(group)),
                "n_cells_with_perturbed_gene": int(group["gene_present"].sum()),
                "fraction_cells_with_perturbed_gene": float(group["gene_present"].mean()),
                "true_mean": float(group["true_value"].mean()),
                "true_median": float(group["true_value"].median()),
                "base_pred_mean": float(group["base_pred_value"].mean()),
                "base_pred_median": float(group["base_pred_value"].median()),
                "perturbed_pred_mean": float(group["perturbed_pred_value"].mean()),
                "perturbed_pred_median": float(group["perturbed_pred_value"].median()),
                "delta_pred_mean": float(group["delta_pred_value"].mean()),
                "delta_pred_median": float(group["delta_pred_value"].median()),
                "delta_pred_abs_median": float(group["delta_pred_value"].abs().median()),
                "embedding_l2_shift_mean": float(group["embedding_l2_shift"].mean()),
                "embedding_l2_shift_median": float(group["embedding_l2_shift"].median()),
                "embedding_cosine_original_perturbed_mean": float(group["embedding_cosine_original_perturbed"].mean()),
                "age_program_contribution_score_mean": float(group["age_program_contribution_score"].mean()),
                "age_program_contribution_score_median": _median(group["age_program_contribution_score"]),
                "age_acceleration_score_mean": float(group["age_acceleration_score"].mean()),
                "age_acceleration_score_median": float(group["age_acceleration_score"].median()),
                "present_delta_pred_mean": float(group.loc[group["gene_present"], "delta_pred_value"].mean())
                if group["gene_present"].any()
                else np.nan,
                "present_embedding_l2_shift_mean": float(group.loc[group["gene_present"], "embedding_l2_shift"].mean())
                if group["gene_present"].any()
                else np.nan,
                "present_embedding_l2_shift_median": float(group.loc[group["gene_present"], "embedding_l2_shift"].median())
                if group["gene_present"].any()
                else np.nan,
                "present_age_program_contribution_score_mean": float(
                    group.loc[group["gene_present"], "age_program_contribution_score"].mean()
                )
                if group["gene_present"].any()
                else np.nan,
                "present_age_program_contribution_score_median": _median(
                    group.loc[group["gene_present"], "age_program_contribution_score"]
                )
                if group["gene_present"].any()
                else np.nan,
                "present_age_acceleration_score_mean": float(
                    group.loc[group["gene_present"], "age_acceleration_score"].mean()
                )
                if group["gene_present"].any()
                else np.nan,
                "present_age_acceleration_score_median": float(
                    group.loc[group["gene_present"], "age_acceleration_score"].median()
                )
                if group["gene_present"].any()
                else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
