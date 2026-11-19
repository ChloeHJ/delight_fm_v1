#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_from_disk


from paths import RUN as RUN_DIR, TASK as TASK_DIR, SRC as SCRIPT_DIR

from geneformer_mamba_model import (  # noqa: E402
    GeneformerMambaForAtlasPretraining,
    _pool_hidden_states,
    _pool_hvg_hidden_states,
)


DEFAULT_MODEL = RUN_DIR / "outputs/model_registry/REQUIRES_EXPLICIT_MODEL"

DEFAULT_TOKEN_DICT = RUN_DIR / "data/reference/token_dictionary.pkl"
DEFAULT_GTF = RUN_DIR / "data/reference/gencode.v44.annotation.gtf.gz"
from lineage_groups import load_original_lineages, grouped_cell_effects, SCORE_DEFINITION
DEFAULT_MARKER_LINEAGE_MAP = {
    "EPCAM": "Epithelium",
    "CDH1": "Epithelium",
    "KRT8": "Epithelium",
    "MUC1": "Epithelium",
    "CHGA": "EECs",
    "NEUROD1": "EECs",
    "NEUROG3": "EECs",
    "DCN": "Fibroblast",
    "LUM": "Fibroblast",
    "COL1A1": "Fibroblast",
    "PDGFRB": "Pericyte",
    "RGS5": "Pericyte",
    "PECAM1": "Endothelium",
    "PTPRC": "T-cell",
    "CD3D": "T-cell",
    "CD3E": "T-cell",
    "MS4A1": "B-cells",
    "CD19": "B-cells",
    "CD68": "Myeloid",
    "CD14": "Myeloid",
    "LYZ": "Myeloid",
}


def parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def normalize_missing(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "na", "n/a", "null"}:
        return None
    return text


def load_json_arg(raw: str | None) -> dict:
    if not raw:
        return {}
    path = Path(raw)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(raw)


def load_marker_lineage_map(raw: str | None) -> dict[str, str]:
    if raw is None or raw.strip().lower() in {"", "default"}:
        return dict(DEFAULT_MARKER_LINEAGE_MAP)
    if raw.strip().lower() in {"none", "off", "0"}:
        return {}
    loaded = load_json_arg(raw)
    return {str(gene): str(lineage) for gene, lineage in loaded.items()}


def normalize_gene_key(value) -> str:
    return str(value).strip().upper()


def marker_lineage_for_gene(gene: dict, marker_lineage_map: dict[str, str]) -> str | None:
    lookup = {normalize_gene_key(key): value for key, value in marker_lineage_map.items()}
    for key in ("input_gene", "ensembl_id"):
        value = gene.get(key)
        if value is not None and normalize_gene_key(value) in lookup:
            return lookup[normalize_gene_key(value)]
    return None


def apply_filters(dataset, filters: dict, nproc: int):
    if not filters:
        return dataset
    for key in filters:
        if key not in dataset.column_names:
            raise ValueError(f"Filter column '{key}' not in dataset columns: {dataset.column_names}")

    allowed = {key: {str(item) for item in values} for key, values in filters.items()}

    def keep(row):
        for key, values in allowed.items():
            if str(row.get(key)) not in values:
                return False
        return True

    return dataset.filter(keep, num_proc=max(1, nproc))


def cap_dataset(dataset, max_cells: int | None, seed: int):
    """Keep all cells for an unset or zero cap; sample only positive caps."""
    if max_cells is None or max_cells == 0 or len(dataset) <= max_cells:
        return dataset
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(dataset), size=max_cells, replace=False))
    return dataset.select(keep.tolist())


def read_gene_file(path: str | None) -> list[str]:
    if not path:
        return []
    genes = []
    with open(path) as handle:
        for line in handle:
            value = line.strip()
            if value and not value.startswith("#"):
                genes.append(value.split()[0])
    return genes


def parse_gtf_symbols(gtf_gz: Path, wanted_symbols: set[str], mapping_tsv: Path | None = None) -> dict[str, str]:
    if not wanted_symbols:
        return {}
    mapping: dict[str, str] = {}
    mapping_tsv = Path(mapping_tsv) if mapping_tsv else RUN_DIR / "data/reference/gene_symbol_to_ensembl.tsv"
    if mapping_tsv.exists():
        reference = pd.read_csv(mapping_tsv, sep="\t")
        for symbol, gene_id in reference.loc[
            reference["gene_symbol"].isin(wanted_symbols), ["gene_symbol", "ensembl_id"]
        ].dropna().itertuples(index=False):
            mapping.setdefault(str(symbol), re.sub(r"\.\d+$", "", str(gene_id)))
    if not gtf_gz.exists() or len(mapping) == len(wanted_symbols):
        return mapping
    attr_re = re.compile(r'(\S+) "([^"]+)"')
    with gzip.open(gtf_gz, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            attrs = {match.group(1): match.group(2) for match in attr_re.finditer(fields[8])}
            symbol = attrs.get("gene_name")
            gene_id = attrs.get("gene_id")
            if symbol in wanted_symbols and gene_id:
                mapping.setdefault(symbol, re.sub(r"\.\d+$", "", gene_id))
                if len(mapping) == len(wanted_symbols):
                    break
    return mapping


def token_for_gene(gene: str, gene_to_token: dict, symbol_to_ensembl: dict[str, str]) -> tuple[str, int | None]:
    ensembl_id = gene
    if gene not in gene_to_token and not gene.startswith("ENSG"):
        ensembl_id = symbol_to_ensembl.get(gene, gene)
    token = gene_to_token.get(ensembl_id) or gene_to_token.get(gene)
    return ensembl_id, None if token is None else int(token)


def load_token_dictionary(path: Path):
    with open(path, "rb") as handle:
        gene_to_token = pickle.load(handle)
    token_to_gene = {int(token): gene for gene, token in gene_to_token.items() if isinstance(token, int)}
    return gene_to_token, token_to_gene


def load_label_maps(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def resolve_genes(args, dataset, gene_to_token: dict, token_to_gene: dict) -> list[dict]:
    requested = parse_csv_list(args.genes) + read_gene_file(args.gene_file)
    if args.genes_to_perturb == "all":
        token_counts = defaultdict(int)
        for ids in dataset["input_ids"]:
            for token in set(ids):
                token_counts[int(token)] += 1
        rows = [
            {
                "input_gene": token_to_gene.get(token, str(token)),
                "ensembl_id": token_to_gene.get(token, str(token)),
                "token": int(token),
                "present_cells": int(count),
            }
            for token, count in token_counts.items()
            if token in token_to_gene
        ]
        rows.sort(key=lambda row: (-row["present_cells"], row["ensembl_id"]))
        return rows[: args.max_genes] if args.max_genes else rows

    if not requested:
        raise ValueError("Provide --genes/--gene-file, or set --genes-to-perturb all.")

    symbols = {gene for gene in requested if not gene.startswith("ENSG") and gene not in gene_to_token}
    symbol_to_ensembl = parse_gtf_symbols(Path(args.gencode_gtf_gz), symbols, getattr(args, "gene_mapping_tsv", None))
    resolved = []
    missing = []
    for gene in requested:
        ensembl_id, token = token_for_gene(gene, gene_to_token, symbol_to_ensembl)
        if token is None:
            missing.append(gene)
            continue
        resolved.append(
            {
                "input_gene": gene,
                "perturbation_label": gene,
                "perturbation_kind": "single_gene",
                "ensembl_id": ensembl_id,
                "token": int(token),
                "tokens": [int(token)],
                "n_tokens": 1,
                "present_cells": None,
            }
        )
    if missing:
        print(f"WARNING: {len(missing)} requested genes were not in the token dictionary: {missing[:20]}")
    if not resolved:
        raise ValueError("None of the requested genes resolved to token dictionary entries.")
    return resolved


def load_regulon_perturbations(args, gene_to_token: dict) -> list[dict]:
    if not args.regulon_tsv:
        return []
    regulon = pd.read_csv(args.regulon_tsv, sep=None, engine="python")
    lower_to_col = {column.lower(): column for column in regulon.columns}
    tf_col = lower_to_col.get("tf") or lower_to_col.get("source") or lower_to_col.get("transcription_factor")
    target_col = lower_to_col.get("target") or lower_to_col.get("target_gene")
    if tf_col is None or target_col is None:
        raise ValueError(f"Regulon TSV must contain tf/source and target columns: {list(regulon.columns)}")
    mor_col = lower_to_col.get("mor") or lower_to_col.get("mode") or lower_to_col.get("effect") or lower_to_col.get("weight")
    confidence_col = lower_to_col.get("confidence")
    name_col = lower_to_col.get("eregulon_name")

    keep_tfs = {normalize_gene_key(tf) for tf in parse_csv_list(args.regulon_tfs)}
    if keep_tfs and "ALL" not in keep_tfs:
        regulon = regulon[regulon[tf_col].map(normalize_gene_key).isin(keep_tfs)]
    keep_confidence = set(parse_csv_list(args.regulon_confidence_levels))
    if keep_confidence and confidence_col:
        regulon = regulon[regulon[confidence_col].astype(str).isin(keep_confidence)]
    if regulon.empty:
        raise ValueError("No regulon rows remain after filtering.")

    symbols = set(regulon[tf_col].dropna().astype(str)) | set(regulon[target_col].dropna().astype(str))
    symbols = {gene for gene in symbols if not gene.startswith("ENSG") and gene not in gene_to_token}
    symbol_to_ensembl = parse_gtf_symbols(Path(args.gencode_gtf_gz), symbols, getattr(args, "gene_mapping_tsv", None))

    perturbations = []
    for name, group in regulon.groupby(name_col or tf_col, sort=True):
        tfs = group[tf_col].dropna().astype(str).unique()
        if len(tfs) != 1:
            raise ValueError(f"Regulon {name} must identify one TF: {tfs}")
        tf = str(tfs[0])
        target_rows = group.dropna(subset=[target_col]).drop_duplicates(subset=[target_col]).copy()
        if mor_col and args.regulon_signed_mode != "all":
            mor = pd.to_numeric(target_rows[mor_col], errors="coerce")
            if args.regulon_signed_mode == "activation":
                target_rows = target_rows[mor > 0]
            elif args.regulon_signed_mode == "repression":
                target_rows = target_rows[mor < 0]
        if args.regulon_max_targets and len(target_rows) > args.regulon_max_targets:
            target_rows = target_rows.head(args.regulon_max_targets)

        requested_genes = target_rows[target_col].dropna().astype(str).tolist()
        if args.regulon_include_tf:
            requested_genes = [tf] + requested_genes

        token_rows = []
        missing = []
        seen_tokens = set()
        for gene in requested_genes:
            ensembl_id, token = token_for_gene(gene, gene_to_token, symbol_to_ensembl)
            if token is None:
                missing.append(gene)
                continue
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            token_rows.append({"gene": gene, "ensembl_id": ensembl_id, "token": int(token)})
        if not token_rows:
            continue
        tokens = [row["token"] for row in token_rows]
        perturbations.append(
            {
                "input_gene": tf,
                "perturbation_label": str(name) if name_col else f"{tf}_regulon",
                "eRegulon_name": str(name) if name_col else None,
                "perturbation_kind": "regulon",
                "regulon_source": ";".join(sorted({compartment for value in group["compartments"].dropna().astype(str) for compartment in value.split(";")})) if "compartments" in group else Path(args.regulon_tsv).stem,
                "regulon_path": str(Path(args.regulon_tsv).resolve()),
                "ensembl_id": token_rows[0]["ensembl_id"],
                "token": int(tokens[0]),
                "tokens": tokens,
                "target_genes": ",".join(row["gene"] for row in token_rows),
                "target_tokens": ",".join(str(token) for token in tokens),
                "n_tokens": len(tokens),
                "n_regulon_rows": int(len(group)),
                "n_missing_regulon_genes": int(len(missing)),
                "missing_regulon_genes": ",".join(missing),
                "present_cells": None,
            }
        )
    if not perturbations:
        raise ValueError("No regulon perturbations resolved to token dictionary entries.")
    return perturbations


class PadCollator:
    def __init__(self, pad_token_id: int, max_length: int):
        self.pad_token_id = int(pad_token_id)
        self.max_length = int(max_length)

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        lengths = [min(len(row["input_ids"]), self.max_length) for row in rows]
        batch_len = max(max(lengths), 1)
        input_ids = []
        attention_mask = []
        expression_values = []
        auxiliary_gene_mask = []
        for row, length in zip(rows, lengths):
            ids = list(row["input_ids"][:length])
            values = list(row.get("expression_values", []))[:length]
            values = values + [0.0] * max(0, length - len(values))
            aux = list(row.get("auxiliary_gene_mask", []))[:length]
            aux = aux + [0] * max(0, length - len(aux))
            pad_len = batch_len - length
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * length + [0] * pad_len)
            expression_values.append(values + [0.0] * pad_len)
            auxiliary_gene_mask.append(aux + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "expression_values": torch.tensor(expression_values, dtype=torch.float32),
            "auxiliary_gene_mask": torch.tensor(auxiliary_gene_mask, dtype=torch.long),
        }


def iter_batches(dataset, batch_size: int):
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        yield start, end, [dataset[i] for i in range(start, end)]


def pooled_backbone(model, batch: dict[str, torch.Tensor], device: torch.device):
    batch = {key: value.to(device) for key, value in batch.items()}
    hidden = model.backbone(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        expression_values=batch.get("expression_values"),
    )
    pooled = _pool_hidden_states(hidden, batch["attention_mask"])
    pcw_score = model.pcw_head(pooled).squeeze(-1) if hasattr(model, "pcw_head") else None
    lineage_probs = None
    if getattr(model, "lineage_head", None) is not None:
        auxiliary_pooled = _pool_hvg_hidden_states(
            hidden,
            batch["attention_mask"],
            batch.get("auxiliary_gene_mask"),
        )
        lineage_probs = torch.softmax(model.lineage_head(auxiliary_pooled), dim=-1)
    return (
        pooled.detach().cpu(),
        None if pcw_score is None else pcw_score.detach().cpu(),
        None if lineage_probs is None else lineage_probs.detach().cpu(),
    )


def embed_dataset(model, dataset, collator, batch_size: int, device: torch.device):
    embs = []
    pcw_scores = []
    lineage_probs = []
    with torch.inference_mode():
        for _, _, rows in iter_batches(dataset, batch_size):
            pooled, pcw_score, lineage_prob = pooled_backbone(model, collator(rows), device)
            embs.append(pooled)
            if pcw_score is not None:
                pcw_scores.append(pcw_score)
            if lineage_prob is not None:
                lineage_probs.append(lineage_prob)
    emb = torch.cat(embs, dim=0)
    pcw = torch.cat(pcw_scores, dim=0).numpy() if pcw_scores else np.full(len(dataset), np.nan)
    lineage = torch.cat(lineage_probs, dim=0).numpy() if lineage_probs else None
    return emb, pcw, lineage


def perturb_rows(rows: list[dict], tokens: list[int], perturb_type: str, overexpress_value: float | None):
    token_set = {int(token) for token in tokens}
    perturbed = []
    for row in rows:
        ids = list(row["input_ids"])
        values = list(row.get("expression_values", []))
        auxiliary = list(row.get("auxiliary_gene_mask", []))
        if "auxiliary_gene_mask" in row and len(auxiliary) != len(ids):
            raise ValueError("auxiliary_gene_mask must align with input_ids before perturbation")
        if len(values) < len(ids):
            values.extend([0.0] * (len(ids) - len(values)))
        if perturb_type == "delete":
            keep = [idx for idx, value in enumerate(ids) if int(value) not in token_set]
            new_ids = [ids[idx] for idx in keep]
            new_values = [values[idx] for idx in keep]
        elif perturb_type == "overexpress":
            keep = [idx for idx, value in enumerate(ids) if int(value) not in token_set]
            new_ids = [int(token) for token in tokens] + [ids[idx] for idx in keep]
            if overexpress_value is None:
                front_value = max(values) if values else 5.0
            else:
                front_value = float(overexpress_value)
            new_values = [front_value] * len(tokens) + [values[idx] for idx in keep]
        else:
            raise ValueError(f"Unsupported perturb_type: {perturb_type}")
        new_row = dict(row)
        new_row["input_ids"] = new_ids
        new_row["expression_values"] = new_values
        if "auxiliary_gene_mask" in row:
            remaining_auxiliary = [auxiliary[idx] for idx in keep]
            if perturb_type == "overexpress":
                token_auxiliary = {int(token): mask for token, mask in zip(ids, auxiliary)}
                new_row["auxiliary_gene_mask"] = [token_auxiliary.get(int(token), 0) for token in tokens] + remaining_auxiliary
            else:
                new_row["auxiliary_gene_mask"] = remaining_auxiliary
        new_row["length"] = len(new_ids)
        perturbed.append(new_row)
    return perturbed


def cosine_to_centroids(embeddings: torch.Tensor, centroids: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    if not centroids:
        return {}
    emb_norm = F.normalize(embeddings, dim=1)
    return {
        label: torch.matmul(emb_norm, F.normalize(centroid, dim=0)).numpy()
        for label, centroid in centroids.items()
    }


def build_centroids(dataset, embeddings: torch.Tensor, state_key: str, min_cells: int):
    if not state_key or state_key not in dataset.column_names:
        return {}
    values = [normalize_missing(value) for value in dataset[state_key]]
    centroids = {}
    for label in sorted({value for value in values if value is not None}):
        idx = [i for i, value in enumerate(values) if value == label]
        if len(idx) >= min_cells:
            centroids[label] = embeddings[idx].mean(dim=0)
    return centroids


def summarize_vector(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_abs_median": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_abs_median": float(np.median(np.abs(finite))),
    }


def numeric_column(dataset, column: str | None) -> np.ndarray | None:
    if not column or column not in dataset.column_names:
        return None
    return pd.to_numeric(pd.Series(dataset[column]), errors="coerce").to_numpy(dtype=float)


def integer_column(dataset, column: str | None) -> np.ndarray | None:
    values = numeric_column(dataset, column)
    if values is None:
        return None
    return np.where(np.isfinite(values), values, -1).astype(int)


def class_probability_for_ids(probabilities: np.ndarray | None, class_ids: np.ndarray | None) -> np.ndarray:
    if probabilities is None or class_ids is None:
        return np.full(0 if class_ids is None else len(class_ids), np.nan)
    scores = np.full(len(class_ids), np.nan, dtype=float)
    valid = (class_ids >= 0) & (class_ids < probabilities.shape[1])
    if valid.any():
        scores[valid] = probabilities[np.where(valid)[0], class_ids[valid]]
    return scores


def class_probability_for_label(
    probabilities: np.ndarray | None,
    lineage_label: str | None,
    lineage_label_to_id: dict[str, int],
    n_cells: int,
) -> np.ndarray:
    if probabilities is None or lineage_label is None or lineage_label not in lineage_label_to_id:
        return np.full(n_cells, np.nan, dtype=float)
    lineage_id = int(lineage_label_to_id[lineage_label])
    if lineage_id < 0 or lineage_id >= probabilities.shape[1]:
        return np.full(n_cells, np.nan, dtype=float)
    return probabilities[:, lineage_id]


def contribution_direction(perturb_type: str) -> float:
    if perturb_type == "delete":
        return -1.0
    if perturb_type == "overexpress":
        return 1.0
    return np.nan


def pcw_bin_column(values: np.ndarray | None) -> list[str] | None:
    if values is None:
        return None
    bins = [0, 8, 12, 16, 20, 24, 32, 40, np.inf]
    labels = ["pcw_0_8", "pcw_8_12", "pcw_12_16", "pcw_16_20", "pcw_20_24", "pcw_24_32", "pcw_32_40", "adult"]
    out = []
    for value in values:
        if not np.isfinite(value):
            out.append(None)
            continue
        idx = int(np.searchsorted(bins, value, side="right") - 1)
        out.append(labels[max(0, min(idx, len(labels) - 1))])
    return out


def write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    columns = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def auto_umap_columns(columns: list[str]) -> tuple[str, str]:
    candidates = [
        ("umap_harmony_1", "umap_harmony_2"),
        ("UMAP_1", "UMAP_2"),
        ("umap_1", "umap_2"),
        ("UMAPHARMONY_1", "UMAPHARMONY_2"),
        ("umap.harmony_1", "umap.harmony_2"),
    ]
    for x_col, y_col in candidates:
        if x_col in columns and y_col in columns:
            return x_col, y_col
    numeric_like = [col for col in columns if col != "cell_id"]
    if len(numeric_like) >= 2:
        return numeric_like[0], numeric_like[1]
    raise ValueError(f"Could not infer UMAP coordinate columns from: {columns}")


def load_umap_coordinates(path: Path | None, cell_ids: list[str]) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    umap = pd.read_csv(path)
    if "cell_id" not in umap.columns:
        raise ValueError(f"UMAP file must contain a cell_id column: {path}")
    x_col, y_col = auto_umap_columns(list(umap.columns))
    aligned = umap[["cell_id", x_col, y_col]].drop_duplicates("cell_id").set_index("cell_id")
    aligned = aligned.reindex([str(cell_id) for cell_id in cell_ids])
    aligned = aligned.rename(columns={x_col: "umap_1", y_col: "umap_2"}).reset_index()
    missing = int(aligned["umap_1"].isna().sum() + aligned["umap_2"].isna().sum())
    if missing:
        print(f"WARNING: {missing // 2} selected cells had no UMAP coordinates in {path}")
    return aligned


def plot_metric_umap(ax, plot_df: pd.DataFrame, metric: str, title: str, cmap: str, center_zero: bool):
    valid = plot_df["umap_1"].notna() & plot_df["umap_2"].notna()
    present = valid & plot_df["gene_present"].astype(bool)
    background = valid & ~plot_df["gene_present"].astype(bool)
    ax.scatter(
        plot_df.loc[background, "umap_1"],
        plot_df.loc[background, "umap_2"],
        s=3,
        c="#d0d0d0",
        alpha=0.35,
        linewidths=0,
    )
    values = plot_df.loc[present, metric].to_numpy(dtype=float)
    if values.size:
        finite = values[np.isfinite(values)]
        if finite.size:
            if center_zero:
                vmax = float(np.nanpercentile(np.abs(finite), 99))
                if vmax == 0:
                    vmax = 1e-9
                vmin = -vmax
            else:
                vmin = float(np.nanpercentile(finite, 1))
                vmax = float(np.nanpercentile(finite, 99))
                if vmin == vmax:
                    vmax = vmin + 1e-9
            sc = ax.scatter(
                plot_df.loc[present, "umap_1"],
                plot_df.loc[present, "umap_2"],
                s=5,
                c=values,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=0.9,
                linewidths=0,
            )
            ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])


def write_umap_plots(
    output_dir: Path,
    output_prefix: str,
    perturb_type: str,
    summary_df: pd.DataFrame,
    plot_payloads: dict[str, pd.DataFrame],
    top_n: int,
) -> list[str]:
    if not plot_payloads:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "umap_perturbation_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    rank_col = "present_embedding_l2_shift_median"
    # Every perturbation retains its numeric UMAP payload, even when only a
    # compact subset receives a figure.
    for label, payload in plot_payloads.items():
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
        payload.to_csv(plots_dir / f"{output_prefix}_{perturb_type}_{safe_label}_umap.csv.gz", index=False)
    ranked = summary_df.sort_values(rank_col, ascending=False, na_position="last")
    if top_n <= 0:
        top_n = len(ranked)
    written = []
    for _, row in ranked.head(top_n).iterrows():
        gene = str(row.get("perturbation_label", row["input_gene"]))
        plot_df = plot_payloads.get(gene)
        if plot_df is None:
            continue
        safe_gene = re.sub(r"[^A-Za-z0-9_.-]+", "_", gene)
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        axes = [ax]
        fig.suptitle(
            f"{gene} {perturb_type}: present={int(row['n_cells_with_gene'])}/{int(row['n_cells'])}",
            fontsize=12,
        )
        plot_metric_umap(axes[0], plot_df, "embedding_l2_shift", f"{gene}\nEmbedding L2 shift", "viridis_r", False)
        png = plots_dir / f"{output_prefix}_{perturb_type}_{safe_gene}_umap.png"
        svg = plots_dir / f"{output_prefix}_{perturb_type}_{safe_gene}_umap.pdf"
        fig.savefig(png, dpi=180)
        fig.savefig(svg)
        plot_df.to_csv(plots_dir / f"{output_prefix}_{perturb_type}_{safe_gene}_umap.csv.gz", index=False)
        plt.close(fig)
        written.extend([str(png), str(svg)])
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Run v9 atlas Mamba in silico gene perturbations on cell embeddings."
    )
    parser.add_argument("--reference-baseline", required=True)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--signed-target-plan")
    parser.add_argument("--signed-target-strengths", default="1")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    parser.add_argument("--train-dataset", default=str(RUN_DIR / "tokenized" / "train.dataset"))
    parser.add_argument("--test-dataset", default=str(RUN_DIR / "tokenized" / "test.dataset"))
    parser.add_argument("--dataset-split", choices=["train", "test", "both"], default="test")
    parser.add_argument("--token-dictionary", default=str(DEFAULT_TOKEN_DICT))
    parser.add_argument("--label-maps", default=str(RUN_DIR / "tokenized" / "label_maps.json"))
    parser.add_argument("--lineage-grouping-csv", default=str(RUN_DIR / "data/reference/original_lineage_by_cell.csv.gz"))
    parser.add_argument("--gencode-gtf-gz", default=str(DEFAULT_GTF))
    parser.add_argument("--gene-mapping-tsv", default=None, help="Optional symbol-to-Ensembl reference; defaults to the run reference.")
    parser.add_argument("--output-dir", default=str(RUN_DIR / "outputs" / "in_silico_perturbation" / "atlas_mamba_l4_d256_lr3e-4" / "epoch_5p0"))
    parser.add_argument("--output-prefix", default="v9_l4_d256_lr3e-4_epoch5")
    parser.add_argument("--umap-csv", default=str(RUN_DIR / "exported" / "umap_harmony.csv.gz"))
    parser.add_argument(
        "--umap-plot-top-genes",
        type=int,
        default=0,
        help="Number of top perturbations to plot on UMAP by embedding shift; <=0 plots all.",
    )
    parser.add_argument(
        "--marker-lineage-map-json",
        default="default",
        help="JSON string/path mapping gene symbols to expected lineage labels; use 'none' to disable.",
    )
    parser.add_argument("--genes", default=None, help="Comma-separated gene symbols or Ensembl IDs.")
    parser.add_argument("--gene-file", default=None, help="One gene symbol or Ensembl ID per line.")
    parser.add_argument("--genes-to-perturb", choices=["list", "all"], default="list")
    parser.add_argument("--max-genes", type=int, default=200, help="Cap for --genes-to-perturb all.")
    parser.add_argument("--regulon-tsv", default=None, help="Optional TSV/CSV with tf/source and target columns.")
    parser.add_argument("--regulon-tfs", default=None, help="Comma-separated TFs to keep from --regulon-tsv.")
    parser.add_argument("--regulon-include-tf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--regulon-max-targets", type=int, default=50)
    parser.add_argument("--regulon-confidence-levels", default=None, help="Optional comma-separated confidence values to keep.")
    parser.add_argument("--regulon-signed-mode", choices=["all", "activation", "repression"], default="all")
    parser.add_argument("--perturb-type", choices=["delete", "overexpress"], default="delete")
    parser.add_argument("--filter-json", default=None, help="JSON string/path, e.g. '{\"Lineage\":[\"Epithelial\"]}'.")
    parser.add_argument("--start-state-key", default="Lineage")
    parser.add_argument("--start-state", default=None)
    parser.add_argument("--goal-states", default=None, help="Comma-separated states in --start-state-key.")
    parser.add_argument("--centroid-keys", default="Lineage,Cluster,AgeGroup")
    parser.add_argument("--pcw-column", default="PCW")
    parser.add_argument("--max-cells", type=int, default=5000, help="Maximum test cells; 0 uses all selected cells.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overexpress-value", type=float, default=None)
    parser.add_argument("--min-centroid-cells", type=int, default=25)
    parser.add_argument("--save-per-cell", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.device == "cpu":
        raise RuntimeError(
            "This v9 Mamba checkpoint requires CUDA for inference because the installed "
            "mamba_ssm kernels do not support CPU forward passes in this environment. "
            "Run through gpu_v9_in_silico_perturbation_1gpu.sh or pass --device cuda on a GPU node."
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false in this process.")

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    if args.dataset_split in {"train", "both"}:
        datasets.append(load_from_disk(args.train_dataset))
    if args.dataset_split in {"test", "both"}:
        datasets.append(load_from_disk(args.test_dataset))
    dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)

    filters = load_json_arg(args.filter_json)
    if args.start_state:
        filters.setdefault(args.start_state_key, [])
        filters[args.start_state_key] = list(dict.fromkeys(list(filters[args.start_state_key]) + [args.start_state]))
    dataset = apply_filters(dataset, filters, args.nproc)
    dataset = cap_dataset(dataset, args.max_cells, args.seed)
    if len(dataset) == 0:
        raise ValueError("No cells remain after filtering/capping.")

    gene_to_token, token_to_gene = load_token_dictionary(Path(args.token_dictionary))
    perturbations = []
    if args.signed_target_plan:
        if args.regulon_tsv or args.genes or args.gene_file or args.genes_to_perturb == 'all':
            raise ValueError('Signed targets are a separate experiment')
        from signed_tf_targets import load_programs
        perturbations.extend(load_programs(args.signed_target_plan, args.signed_target_strengths))
        if perturbations[0]['signed_target_plan'].get('signed_edit_version') != 'float32_v2':
            raise ValueError('Corrected float32_v2 signed plan required')
        if perturbations[0]['signed_target_plan']['perturb_type'] != args.perturb_type:
            raise ValueError('Signed plan perturbation mode mismatch')
        for program in perturbations:
            program.setdefault('ensembl_id', '')
    if args.regulon_tsv:
        perturbations.extend(load_regulon_perturbations(args, gene_to_token))
    if args.genes or args.gene_file or args.genes_to_perturb == "all":
        perturbations.extend(resolve_genes(args, dataset, gene_to_token, token_to_gene))
    if not perturbations and not args.baseline_only:
        raise ValueError("Provide --genes/--gene-file, --genes-to-perturb all, or --regulon-tsv.")
    pd.DataFrame(perturbations).to_csv(output_dir / f"{args.output_prefix}_resolved_genes.csv", index=False)

    cell_ids = [str(value) for value in dataset["cell_id"]]
    if "OriginalLineage" in dataset.column_names:
        original_lineages = np.asarray(dataset["OriginalLineage"], dtype=str)
        if any(value in {"", "None", "nan"} for value in original_lineages):
            raise ValueError("OriginalLineage contains missing annotations")
    else:
        original_lineages = load_original_lineages(args.lineage_grouping_csv, cell_ids)
    from baseline_contract import strict_classifier, classification_baseline
    model = strict_classifier(args.model_dir)
    model.eval()
    device = torch.device(args.device)
    model.to(device)
    collator = PadCollator(pad_token_id=int(model.config.pad_token_id), max_length=int(model.config.max_position_embeddings))

    base_emb, base_pcw_score, base_lineage_probs = embed_dataset(model, dataset, collator, args.batch_size, device)
    classification_baseline(args.reference_baseline, cell_ids, base_emb.numpy(),
        base_pcw_score, base_lineage_probs, write=args.baseline_only and not Path(args.reference_baseline).exists())
    if args.baseline_only:
        return
    lineage_ids = integer_column(dataset, "Lineage_id")
    base_lineage_score = class_probability_for_ids(base_lineage_probs, lineage_ids)
    label_maps = load_label_maps(Path(args.label_maps))
    lineage_id_to_label = {
        int(idx): label for label, idx in label_maps.get("Lineage", {}).items()
    }
    lineage_label_to_id = {
        label: int(idx) for label, idx in label_maps.get("Lineage", {}).items()
    }
    marker_lineage_map = load_marker_lineage_map(args.marker_lineage_map_json)
    pd.DataFrame(
        [{"input_gene": gene, "marker_lineage": lineage} for gene, lineage in sorted(marker_lineage_map.items())]
    ).to_csv(output_dir / f"{args.output_prefix}_marker_lineage_map.csv", index=False)
    centroid_keys = parse_csv_list(args.centroid_keys)
    pcw_values = numeric_column(dataset, args.pcw_column)
    pcw_bins = pcw_bin_column(pcw_values)
    centroids_by_key = {
        key: build_centroids(dataset, base_emb, key, args.min_centroid_cells)
        for key in centroid_keys
    }
    if pcw_bins is not None:
        tmp = dataset.add_column("_pcw_bin", pcw_bins)
        centroids_by_key["pcw_bin"] = build_centroids(tmp, base_emb, "_pcw_bin", args.min_centroid_cells)

    base_cos = {
        key: cosine_to_centroids(base_emb, centroids)
        for key, centroids in centroids_by_key.items()
    }
    goal_states = parse_csv_list(args.goal_states)
    if args.start_state and not goal_states:
        goal_states = [
            label for label in centroids_by_key.get(args.start_state_key, {})
            if label != args.start_state
        ]

    summary_rows = []
    lineage_score_rows = []
    lineage_group_rows = []
    per_cell_rows = []
    metadata_columns = [
        column for column in ["cell_id", "Lineage", "Cluster", "AgeGroup", "PCW", "sample_id", "Location"]
        if column in dataset.column_names
    ]
    umap_coords = load_umap_coordinates(Path(args.umap_csv), cell_ids)
    plot_payloads = {}

    for gene in perturbations:
        tokens = [int(token) for token in gene.get("tokens", [gene["token"]])]
        token = int(tokens[0])
        perturbation_label = gene.get("perturbation_label", gene["input_gene"])
        marker_lineage = marker_lineage_for_gene(gene, marker_lineage_map)
        gene_rows = []
        perturbed_embs = []
        perturbed_pcw_scores = []
        perturbed_lineage_probs = []
        present_mask = []
        with torch.inference_mode():
            for start, _, rows in iter_batches(dataset, args.batch_size):
                batch_present = [bool({int(x) for x in row["input_ids"]} & set(tokens)) for row in rows]
                if gene.get('signed_target_plan') is not None:
                    from signed_tf_targets import edit_signed_rows
                    perturbed, _ = edit_signed_rows(rows, gene['signed_target_plan'], gene['signed_strength'])
                else:
                    perturbed = perturb_rows(rows, tokens, args.perturb_type, args.overexpress_value)
                pooled, pcw_score, lineage_prob = pooled_backbone(model, collator(perturbed), device)
                perturbed_embs.append(pooled)
                if pcw_score is not None:
                    perturbed_pcw_scores.append(pcw_score)
                if lineage_prob is not None:
                    perturbed_lineage_probs.append(lineage_prob)
                present_mask.extend(batch_present)
        pert_emb = torch.cat(perturbed_embs, dim=0)
        pert_pcw_score = torch.cat(perturbed_pcw_scores, dim=0).numpy() if perturbed_pcw_scores else np.full(len(dataset), np.nan)
        pert_lineage_probs = torch.cat(perturbed_lineage_probs, dim=0).numpy() if perturbed_lineage_probs else None
        delta = pert_emb - base_emb
        l2_shift = torch.linalg.vector_norm(delta, dim=1).numpy()
        cosine_original_perturbed = F.cosine_similarity(base_emb, pert_emb, dim=1).numpy()
        delta_pcw_score = pert_pcw_score - base_pcw_score
        pert_lineage_score = class_probability_for_ids(pert_lineage_probs, lineage_ids)
        delta_lineage_score = pert_lineage_score - base_lineage_score
        base_marker_lineage_score = class_probability_for_label(
            base_lineage_probs,
            marker_lineage,
            lineage_label_to_id,
            len(dataset),
        )
        pert_marker_lineage_score = class_probability_for_label(
            pert_lineage_probs,
            marker_lineage,
            lineage_label_to_id,
            len(dataset),
        )
        delta_marker_lineage_score = pert_marker_lineage_score - base_marker_lineage_score
        lineage_contribution_score = contribution_direction(args.perturb_type) * delta_marker_lineage_score
        present = np.asarray(present_mask, dtype=bool)

        row = {
            **gene,
            "perturb_type": args.perturb_type,
            "perturbation_label": perturbation_label,
            "perturbation_kind": gene.get("perturbation_kind", "single_gene"),
            "n_cells": int(len(dataset)),
            "n_cells_with_gene": int(present.sum()),
            "fraction_cells_with_gene": float(present.mean()),
            "embedding_l2_shift_mean": float(np.mean(l2_shift)),
            "embedding_l2_shift_median": float(np.median(l2_shift)),
            "embedding_cosine_original_perturbed_mean": float(np.mean(cosine_original_perturbed)),
            **summarize_vector(delta_pcw_score, "delta_pcw_score"),
            **summarize_vector(delta_lineage_score, "delta_lineage_score"),
            "marker_lineage": marker_lineage,
            **summarize_vector(delta_marker_lineage_score, "delta_marker_lineage_score"),
            **summarize_vector(lineage_contribution_score, "lineage_contribution_score"),
        }
        if present.any():
            row.update(summarize_vector(l2_shift[present], "present_embedding_l2_shift"))
            row.update(summarize_vector(delta_pcw_score[present], "present_delta_pcw_score"))
            row.update(summarize_vector(delta_lineage_score[present], "present_delta_lineage_score"))
            row.update(summarize_vector(delta_marker_lineage_score[present], "present_delta_marker_lineage_score"))
            row.update(summarize_vector(lineage_contribution_score[present], "present_lineage_contribution_score"))
        if base_lineage_probs is not None and pert_lineage_probs is not None:
            delta_lineage_probs = pert_lineage_probs - base_lineage_probs
            for lineage_id, label in sorted(lineage_id_to_label.items()):
                if lineage_id < delta_lineage_probs.shape[1]:
                    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
                    lineage_delta = delta_lineage_probs[:, lineage_id]
                    lineage_contribution = contribution_direction(args.perturb_type) * lineage_delta
                    row[f"delta_lineage_prob_{safe_label}_mean"] = float(np.nanmean(lineage_delta))
                    lineage_score_row = {
                        **gene,
                        "perturb_type": args.perturb_type,
                        "lineage": label,
                        "is_marker_lineage": bool(marker_lineage == label),
                        "n_cells": int(len(dataset)),
                        "n_cells_with_gene": int(present.sum()),
                        "fraction_cells_with_gene": float(present.mean()),
                        **summarize_vector(lineage_delta, "delta_lineage_prob"),
                        **summarize_vector(lineage_contribution, "lineage_contribution_score"),
                    }
                    if present.any():
                        lineage_score_row.update(
                            summarize_vector(lineage_delta[present], "present_delta_lineage_prob")
                        )
                        lineage_score_row.update(
                            summarize_vector(
                                lineage_contribution[present],
                                "present_lineage_contribution_score",
                            )
                        )
                    lineage_score_rows.append(lineage_score_row)

        lineage_group_rows.extend(grouped_cell_effects(gene, original_lineages, present,
            delta_lineage_score, l2_shift, args.perturb_type))
        pert_cos_by_key = {
            key: cosine_to_centroids(pert_emb, centroids)
            for key, centroids in centroids_by_key.items()
        }
        for key, state_cos in pert_cos_by_key.items():
            for label, pert_values in state_cos.items():
                delta_cos = pert_values - base_cos[key][label]
                row[f"delta_cosine_to_{key}_{label}_mean"] = float(np.nanmean(delta_cos))
                if key == args.start_state_key and label in goal_states:
                    row[f"goal_delta_cosine_to_{label}_mean"] = float(np.nanmean(delta_cos))
        summary_rows.append(row)

        if args.save_per_cell:
            for idx in range(len(dataset)):
                cell = {column: dataset[idx].get(column) for column in metadata_columns}
                gene_rows.append(
                    {
                        **cell,
                        "OriginalLineage": str(original_lineages[idx]),
                        "input_gene": gene["input_gene"],
                        "perturbation_label": gene.get("perturbation_label", gene["input_gene"]),
                        "perturbation_kind": gene.get("perturbation_kind", "single_gene"),
                        "ensembl_id": gene["ensembl_id"],
                        "token": token,
                        "target_tokens": ",".join(str(token) for token in tokens),
                        "gene_present": bool(present[idx]),
                        "embedding_l2_shift": float(l2_shift[idx]),
                        "embedding_cosine_original_perturbed": float(cosine_original_perturbed[idx]),
                        "base_pcw_score": float(base_pcw_score[idx]),
                        "perturbed_pcw_score": float(pert_pcw_score[idx]),
                        "delta_pcw_score": float(delta_pcw_score[idx]),
                        "base_lineage_score": float(base_lineage_score[idx]),
                        "perturbed_lineage_score": float(pert_lineage_score[idx]),
                        "delta_lineage_score": float(delta_lineage_score[idx]),
                        "marker_lineage": marker_lineage,
                        "base_marker_lineage_score": float(base_marker_lineage_score[idx]),
                        "perturbed_marker_lineage_score": float(pert_marker_lineage_score[idx]),
                        "delta_marker_lineage_score": float(delta_marker_lineage_score[idx]),
                        "lineage_contribution_score": float(lineage_contribution_score[idx]),
                    }
                )
            per_cell_rows.extend(gene_rows)
        if umap_coords is not None:
            plot_df = umap_coords.copy()
            plot_df["gene_present"] = present
            plot_df["OriginalLineage"] = original_lineages
            plot_df["embedding_l2_shift"] = l2_shift
            plot_df["delta_pcw_score"] = delta_pcw_score
            plot_df["delta_lineage_score"] = delta_lineage_score
            plot_df["lineage_contribution_score"] = lineage_contribution_score
            plot_payloads[perturbation_label] = plot_df
        print(
            f"{perturbation_label} tokens={len(tokens)} first_token={token}: n_present={present.sum()} "
            f"median_l2={np.median(l2_shift):.4f} median_delta_pcw={np.nanmedian(delta_pcw_score):.4f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / f"{args.output_prefix}_{args.perturb_type}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    lineage_scores_path = output_dir / f"{args.output_prefix}_{args.perturb_type}_lineage_scores.csv"
    pd.DataFrame(lineage_score_rows).to_csv(lineage_scores_path, index=False)
    lineage_group_scores_path = output_dir / f"{args.output_prefix}_{args.perturb_type}_lineage_group_scores.csv"
    pd.DataFrame(lineage_group_rows).to_csv(lineage_group_scores_path, index=False)
    if args.save_per_cell:
        write_rows(output_dir / f"{args.output_prefix}_{args.perturb_type}_per_cell.csv", per_cell_rows)
    umap_plot_files = write_umap_plots(
        output_dir,
        args.output_prefix,
        args.perturb_type,
        summary_df,
        plot_payloads,
        args.umap_plot_top_genes,
    )

    manifest = {
        "reference_baseline": args.reference_baseline,
        "baseline_verified": True,
        "training_skipped": True,
        "signed_target_plan": args.signed_target_plan,
        "signed_edit_version": "float32_v2" if args.signed_target_plan else None,
        "inference_precision": "float32; autocast off; TF32 off",
        "auxiliary_mask_aligned": True,
        "lineage_grouping": "OriginalLineage",
        "lineage_group_score_definition": SCORE_DEFINITION,
        "lineage_group_scores_csv": str(lineage_group_scores_path),
        "lineage_grouping_csv": str(Path(args.lineage_grouping_csv).resolve()) if Path(args.lineage_grouping_csv).exists() else None,
        "model_dir": str(Path(args.model_dir).resolve()),
        "dataset_split": args.dataset_split,
        "regulon_tsv": str(Path(args.regulon_tsv).resolve()) if args.regulon_tsv else None,
        "regulon_tfs": parse_csv_list(args.regulon_tfs),
        "regulon_include_tf": args.regulon_include_tf,
        "regulon_max_targets": args.regulon_max_targets,
        "n_cells": len(dataset),
        "filters": filters,
        "perturb_type": args.perturb_type,
        "summary_csv": str(summary_path),
        "lineage_scores_csv": str(lineage_scores_path),
        "resolved_genes_csv": str(output_dir / f"{args.output_prefix}_resolved_genes.csv"),
        "label_maps": str(Path(args.label_maps).resolve()) if Path(args.label_maps).exists() else None,
        "marker_lineage_map_csv": str(output_dir / f"{args.output_prefix}_marker_lineage_map.csv"),
        "umap_csv": str(Path(args.umap_csv).resolve()) if Path(args.umap_csv).exists() else None,
        "umap_plot_files": umap_plot_files,
        "notes": [
            "pcw_score is the pretrained v9 auxiliary PCW head output on the learned scaled target, not raw PCW.",
            "delta_lineage_score is the change in lineage-head softmax probability for each cell's annotated Lineage_id.",
            "delta_marker_lineage_score is the change in lineage-head probability for the gene's expected marker lineage.",
            "lineage_scores_csv reports delta_lineage_prob and signed lineage_contribution_score for every gene-lineage pair.",
            "lineage_contribution_score is signed so deletion-driven decreases, or overexpression-driven increases, in marker-lineage probability are positive.",
            "centroid cosine deltas use original unperturbed cell embeddings as state centroids.",
            "UMAP plots use original Seurat coordinates when --umap-csv is available; colors show perturbation effects, not new UMAP embeddings.",
        ],
    }
    (output_dir / f"{args.output_prefix}_{args.perturb_type}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
