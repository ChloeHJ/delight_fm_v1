"""Donor scatter conventions from the v10 age-perturbation figures."""
import math
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

LOCATION_COLOURS = {"Ileum": "#D4A373", "Colon": "#35858c", "Unspecified": "#808080"}


def regression_plot_data(source):
    """Plot one point per donor/location, including donors sampled at both sites."""
    source = Path(source)
    path = source / "eval_predictions.csv"
    if path.exists():
        columns = set(pd.read_csv(path, nrows=0).columns)
        if {"sample_id", "Location", "true_value", "pred_value"} <= columns:
            cells = pd.read_csv(path, usecols=["sample_id", "Location", "true_value", "pred_value"])
            result = cells.groupby(["sample_id", "Location"], dropna=False).agg(
                n_cells=("true_value", "size"), true_mean=("true_value", "mean"),
                true_median=("true_value", "median"), pred_mean=("pred_value", "mean"),
                pred_median=("pred_value", "median")).reset_index()
        else:
            result = pd.read_csv(source / "donor_level_predictions.csv")
    else:
        result = pd.read_csv(source / "donor_level_predictions.csv")
    if "Location" not in result:
        result["Location"] = "Unspecified"
    result["plot_location_colour"] = result.Location.map(LOCATION_COLOURS).fillna("#808080")
    result["plot_point_area"] = point_size(result.n_cells) if "n_cells" in result else 14
    return result


def location_scatter(ax, frame, x, y):
    for location, part in frame.groupby("Location", dropna=False):
        ax.scatter(part[x], part[y], s=part.plot_point_area,
                   color=LOCATION_COLOURS.get(location, "#808080"), label=str(location),
                   alpha=.8, edgecolors="#333333", linewidths=.25)
    ax.legend(title="Location", fontsize=8, title_fontsize=8, frameon=False)


def point_size(n_cells, scale_max=None):
    """Map counts to marker area, optionally using a figure-specific maximum."""
    counts = pd.to_numeric(n_cells, errors="coerce")
    if scale_max is not None:
        return 14 + 81 * np.clip(counts / scale_max, 0, 1)
    return np.clip(counts / 24, 14, 95)


def cell_count_scale(n_cells):
    """Round the observed maximum up to a readable 1/2/5 scale."""
    counts = pd.to_numeric(n_cells, errors="coerce")
    maximum = max(float(np.nanmax(counts)), 1.)
    unit = 10 ** math.floor(math.log10(maximum))
    return next(multiplier * unit for multiplier in (1, 2, 5, 10) if multiplier * unit >= maximum)


def cell_count_legend(fig, ax, count_label="Cells per donor", scale_max=None):
    references = [100, 500, 1000] if scale_max is None else sorted(set(max(1, round(scale_max * fraction)) for fraction in (.2, .5, 1)))
    handles = [ax.scatter([], [], s=point_size(n, scale_max), facecolor="white", edgecolor="#333333", lw=.4) for n in references]
    fig.legend(handles, [f"{n:,}" for n in references], title=count_label + " (dot size)",
               loc="outside lower center", ncol=3, fontsize=8, title_fontsize=8, frameon=False)


def donor_grid(donors, metric, labels, title, columns=4, shared_y=False):
    values = donors[donors.perturbation_label.isin(labels)].copy()
    values[metric] = pd.to_numeric(values[metric], errors="coerce")
    values = values[np.isfinite(values[metric]) & np.isfinite(values.true_mean)].copy()
    if values.empty:
        return None, values
    if values.duplicated(["perturbation_label", "sample_id"]).any():
        raise ValueError("Donor plots require one row per donor and perturbation")
    scale_max = cell_count_scale(values.n_cells)
    values["plot_point_area"] = point_size(values.n_cells, scale_max)
    values["plot_colour_fraction"] = values.fraction_cells_with_perturbed_gene
    ncols = min(columns, len(labels))
    nrows = math.ceil(len(labels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(.8 * 3.6 * ncols, .8 * (3.3 * nrows + .45)),
                             squeeze=False, sharex=True, layout="constrained")
    for ax, label in zip(axes.flat, labels):
        part = values[values.perturbation_label == label]
        sc = ax.scatter(part.true_mean, part[metric], c=part.plot_colour_fraction,
            s=part.plot_point_area, cmap="Purples", vmin=0, vmax=1, alpha=.8,
            linewidths=.2, edgecolors="#333333")
        ax.axhline(0, color="#333333", lw=.8)
        ax.set_xlabel("Donor true mean log2(PCW)")
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_title(textwrap.fill(label.replace("_", " "), 30), fontsize=9)
        ax.grid(alpha=.2, lw=.5)
        # Retain every point; percentile limits in older plots could clip donors.
        limits = values[metric] if shared_y else part[metric]
        if not limits.empty:
            lo, hi = float(limits.min()), float(limits.max())
            if "l2" in metric:
                ax.set_ylim(0, max(hi * 1.15, 1e-6))
            else:
                span = max(abs(lo), abs(hi), 1e-6) * 1.15
                ax.set_ylim(-span, span)
    for ax in axes.flat[len(labels):]:
        ax.set_visible(False)
    ylabel = {
        "delta_pred_mean": "Mean prediction change · log2(PCW)\n(all cells)",
        "present_delta_pred_mean": "Mean prediction change · log2(PCW)\n(gene-present cells)",
        "age_program_contribution_score_mean": "Age-program contribution (all cells)",
        "present_age_program_contribution_score_mean": "Age-program contribution (gene-present cells)",
        "present_embedding_l2_shift_mean": "Mean embedding L2 shift (gene-present cells)",
    }.get(metric, metric.replace("_", " "))
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.colorbar(sc, ax=list(axes.flat[:len(labels)]), label="Fraction of donor cells with perturbed gene(s)",
                 fraction=.025, pad=.02)
    cell_count_legend(fig, axes.flat[0], scale_max=scale_max)
    fig.suptitle(title, fontsize=11)
    return fig, values
