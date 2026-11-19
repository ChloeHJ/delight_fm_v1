#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_from_disk
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import Trainer, TrainingArguments, set_seed


from paths import RUN as RUN_DIR, TASK as TASK_DIR, SRC as SCRIPT_DIR
from donor_scores import donor_summary

from geneformer_mamba_model import GeneformerMambaForSequenceClassification, _pool_hidden_states  # noqa: E402
from perturb_lineage import (  # noqa: E402
    DEFAULT_GTF,
    DEFAULT_TOKEN_DICT,
    PadCollator,
    apply_filters,
    cap_dataset,
    iter_batches,
    load_json_arg,
    load_umap_coordinates,
    load_regulon_perturbations,
    load_token_dictionary,
    parse_csv_list,
    parse_gtf_symbols,
    pcw_bin_column,
    perturb_rows,
    read_gene_file,
    token_for_gene,
    write_rows,
)


DEFAULT_MODEL_KEY = "selected_age_model"
DEFAULT_EPOCH_LABEL = "selected"
DEFAULT_MODEL = RUN_DIR / "outputs/model_registry/REQUIRES_EXPLICIT_MODEL"

DEFAULT_ADULT_GENES = [
    "GUCA2A",
    "DPEP1",
    "HLA-DRB1",
    "HLA-DQB1",
    "MS4A10",
    "CD74",
    "HLA-DRA",
    "ADIRF",
    "GUCA2B",
    "CYB5A",
]
DEFAULT_FETAL_GENES = [
    "CLDN6",
    "FGG",
    "AMHR2",
    "GAMT",
    "COL6A1",
    "UPK3A",
    "CYP2W1",
    "MXRA8",
    "APOE",
    "CKB",
    "AFP",
    "HPN",
    "TTR",
]
DEFAULT_EARLY_LIFE_GENES = [
    "ARID3A",
    "GDF15",
    "EREG",
    "NAV3",
    "GAS2L3",
    "CA4",
    "BHLHE40",
    "ANKRD22",
    "CXCL2",
]
DEFAULT_TFS = ["all"]
DEFAULT_SCENICPLUS_REGULON_DIR = RUN_DIR / "data" / "scenic" / "epi.ileum"
DEFAULT_GENE_MAPPING_TSV = RUN_DIR / "data/reference/gene_symbol_to_ensembl.tsv"
SCENICPLUS_REGULON_ACTIVITIES = ["+_+", "+_-", "-_+", "-_-"]


class RegressionCollator(PadCollator):
    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        batch = super().__call__(rows)
        if "labels" in rows[0]:
            batch["labels"] = torch.tensor([float(row["labels"]) for row in rows], dtype=torch.float32)
        return batch


def split_dataset(train_path: str, test_path: str, split: str):
    datasets = []
    if split in {"train", "both"}:
        datasets.append(load_from_disk(train_path))
    if split in {"test", "both"}:
        datasets.append(load_from_disk(test_path))
    return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)


def add_numeric_labels(dataset, label_column: str):
    if label_column not in dataset.column_names:
        raise ValueError(f"Missing label column {label_column}; available columns: {dataset.column_names}")

    def encode(batch):
        values = pd.to_numeric(pd.Series(batch[label_column]), errors="coerce")
        batch["labels"] = [None if pd.isna(value) else float(value) for value in values]
        return batch

    return dataset.map(encode, batched=True).filter(lambda row: row["labels"] is not None)


def build_training_args(args, output_dir: Path):
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.downstream_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_strategy": "no",
        "report_to": [],
        "seed": args.seed,
        "remove_unused_columns": False,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    kwargs["eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def compute_regression_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.squeeze(predictions).astype(float)
    labels = np.asarray(labels).astype(float)
    mse = mean_squared_error(labels, preds)
    pearson = np.corrcoef(labels, preds)[0, 1] if len(labels) > 1 else np.nan
    return {
        "mae": mean_absolute_error(labels, preds),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(labels, preds),
        "pearson": float(pearson) if np.isfinite(pearson) else np.nan,
    }


def predict_rows(model, rows: list[dict], collator, device: torch.device) -> np.ndarray:
    batch = collator(rows)
    batch.pop("labels", None)
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        output = model(**batch)
    return output.logits.squeeze(-1).detach().cpu().numpy().astype(float)


def predict_and_embed_rows(model, rows: list[dict], collator, device: torch.device) -> tuple[np.ndarray, torch.Tensor]:
    batch = collator(rows)
    batch.pop("labels", None)
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        hidden = model.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            expression_values=batch.get("expression_values"),
        )
        pooled = _pool_hidden_states(hidden, batch["attention_mask"])
        logits = model.classifier(model.dropout(pooled))
    return logits.squeeze(-1).detach().cpu().numpy().astype(float), pooled.detach().cpu()


def predict_dataset(model, dataset, collator, batch_size: int, device: torch.device) -> np.ndarray:
    preds = []
    for _, _, rows in iter_batches(dataset, batch_size):
        preds.append(predict_rows(model, rows, collator, device))
    return np.concatenate(preds, axis=0)


def predict_and_embed_dataset(model, dataset, collator, batch_size: int, device: torch.device) -> tuple[np.ndarray, torch.Tensor]:
    preds = []
    embs = []
    for _, _, rows in iter_batches(dataset, batch_size):
        batch_preds, batch_embs = predict_and_embed_rows(model, rows, collator, device)
        preds.append(batch_preds)
        embs.append(batch_embs)
    return np.concatenate(preds, axis=0), torch.cat(embs, dim=0)


def resolve_gene_tokens(label: str, genes: list[str], gene_to_token: dict, symbol_to_ensembl: dict[str, str]) -> dict:
    token_rows = []
    missing = []
    seen = set()
    for gene in genes:
        ensembl_id, token = token_for_gene(gene, gene_to_token, symbol_to_ensembl)
        if token is None:
            missing.append(gene)
            continue
        if token in seen:
            continue
        seen.add(token)
        token_rows.append({"gene": gene, "ensembl_id": ensembl_id, "token": int(token)})
    if not token_rows:
        raise ValueError(f"No genes resolved for perturbation {label}; missing={missing}")
    return {
        "input_gene": label,
        "perturbation_label": label,
        "perturbation_kind": "gene_set",
        "ensembl_id": token_rows[0]["ensembl_id"],
        "token": int(token_rows[0]["token"]),
        "tokens": [row["token"] for row in token_rows],
        "target_genes": ",".join(row["gene"] for row in token_rows),
        "target_tokens": ",".join(str(row["token"]) for row in token_rows),
        "n_tokens": len(token_rows),
        "n_missing_genes": len(missing),
        "missing_genes": ",".join(missing),
    }


def load_symbol_to_ensembl(args, wanted_symbols: set[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    mapping_tsv = Path(args.gene_mapping_tsv) if args.gene_mapping_tsv else None
    if mapping_tsv and mapping_tsv.exists() and wanted_symbols:
        df = pd.read_csv(mapping_tsv, sep=None, engine="python")
        lower_to_col = {column.lower(): column for column in df.columns}
        symbol_col = lower_to_col.get("gene_symbol") or lower_to_col.get("symbol") or lower_to_col.get("gene")
        ensembl_col = lower_to_col.get("ensembl_id") or lower_to_col.get("ensembl") or lower_to_col.get("gene_id")
        if symbol_col is None or ensembl_col is None:
            raise ValueError(f"Gene mapping TSV must contain symbol and Ensembl columns: {mapping_tsv}")
        keep = df[symbol_col].astype(str).isin(wanted_symbols)
        for symbol, ensembl_id in df.loc[keep, [symbol_col, ensembl_col]].dropna().itertuples(index=False):
            mapping.setdefault(str(symbol), re.sub(r"\.\d+$", "", str(ensembl_id)))
        missing = {symbol for symbol in wanted_symbols if symbol not in mapping}
    else:
        missing = set(wanted_symbols)
    if missing:
        mapping.update(parse_gtf_symbols(Path(args.gencode_gtf_gz), missing))
    return mapping


def build_gene_set_perturbations(args, gene_to_token: dict) -> list[dict]:
    gene_sets = {
        "adult_genes": parse_csv_list(args.adult_genes),
        "fetal_genes": parse_csv_list(args.fetal_genes),
        "early_life_genes": parse_csv_list(args.early_life_genes),
    }
    all_symbols = {gene for genes in gene_sets.values() for gene in genes if not gene.startswith("ENSG")}
    if args.include_individual_genes:
        all_symbols |= {gene for gene in parse_csv_list(args.genes) if not gene.startswith("ENSG")}
    symbol_to_ensembl = load_symbol_to_ensembl(args, all_symbols)

    perturbations = [
        resolve_gene_tokens(label, genes, gene_to_token, symbol_to_ensembl)
        for label, genes in gene_sets.items()
        if genes
    ]

    if args.include_individual_genes:
        individual = []
        seen = set()
        for genes in gene_sets.values():
            individual.extend(genes)
        individual.extend(parse_csv_list(args.genes))
        individual.extend(read_gene_file(args.gene_file))
        for gene in individual:
            if gene in seen:
                continue
            seen.add(gene)
            try:
                row = resolve_gene_tokens(gene, [gene], gene_to_token, symbol_to_ensembl)
            except ValueError:
                print(f"WARNING: individual gene did not resolve and will be skipped: {gene}")
                continue
            row["perturbation_kind"] = "single_gene"
            perturbations.append(row)
    return perturbations


def load_scenicplus_regulon_perturbations(args, gene_to_token: dict) -> list[dict]:
    activities = parse_csv_list(args.scenicplus_regulon_activities)
    if not activities or any(activity.lower() == "all" for activity in activities):
        activities = list(SCENICPLUS_REGULON_ACTIVITIES)
    scenic_dir = Path(args.scenicplus_regulon_dir)
    if not scenic_dir.exists():
        raise FileNotFoundError(f"ScenicPlus regulon directory not found: {scenic_dir}")
    tfs = parse_csv_list(args.scenicplus_regulon_tfs)
    if any(tf.lower() == "all" for tf in tfs):
        activity_set = set(activities)
        tfs = sorted(
            {
                child.name.split("_direct_", 1)[0]
                for child in scenic_dir.iterdir()
                if child.is_dir()
                and "_direct_" in child.name
                and child.name.rsplit("_direct_", 1)[-1] in activity_set
                and (child / "regulon_target_genes_and_regions.tsv").exists()
            }
        )
    if not tfs:
        return []

    selected: list[tuple[str, str, Path, pd.DataFrame]] = []
    all_symbols = set()
    for tf in tfs:
        tf = str(tf).strip()
        tf_selected = 0
        for suffix in activities:
            path = scenic_dir / f"{tf}_direct_{suffix}" / "regulon_target_genes_and_regions.tsv"
            if not path.exists():
                continue
            df = pd.read_csv(path, sep="\t")
            if "Gene" not in df.columns:
                raise ValueError(f"ScenicPlus file lacks Gene column: {path}")
            genes = [gene for gene in df["Gene"].dropna().astype(str).unique() if gene]
            if genes:
                selected.append((tf, suffix, path, df))
                tf_selected += 1
                all_symbols.update(genes)
                if args.scenicplus_regulon_include_tf:
                    all_symbols.add(tf)
        if tf_selected == 0:
            raise FileNotFoundError(
                f"No nonempty ScenicPlus regulon found for {tf}; checked "
                + ", ".join(f"{tf}_direct_{suffix}" for suffix in activities)
            )

    symbols = {gene for gene in all_symbols if not gene.startswith("ENSG") and gene not in gene_to_token}
    symbol_to_ensembl = load_symbol_to_ensembl(args, symbols)

    perturbations = []
    for tf, suffix, path, df in selected:
        genes = [gene for gene in df["Gene"].dropna().astype(str).drop_duplicates().tolist() if gene]
        if args.scenicplus_regulon_max_targets and len(genes) > args.scenicplus_regulon_max_targets:
            genes = genes[: args.scenicplus_regulon_max_targets]
        requested_genes = [tf] + genes if args.scenicplus_regulon_include_tf else genes

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
            print(f"WARNING: no ScenicPlus regulon genes resolved for {tf}; skipped")
            continue
        tokens = [row["token"] for row in token_rows]
        perturbations.append(
            {
                "input_gene": tf,
                "perturbation_label": f"{tf}_scenicplus_direct_{suffix}_regulon",
                "perturbation_kind": "scenicplus_regulon",
                "regulon_source": scenic_dir.name,
                "scenicplus_regulon_name": f"{tf}_direct_{suffix}",
                "scenicplus_regulon_activity": suffix,
                "scenicplus_regulon_path": str(path),
                "ensembl_id": token_rows[0]["ensembl_id"],
                "token": int(tokens[0]),
                "tokens": tokens,
                "target_genes": ",".join(row["gene"] for row in token_rows),
                "target_tokens": ",".join(str(token) for token in tokens),
                "n_tokens": len(tokens),
                "n_regulon_rows": int(len(df)),
                "n_missing_regulon_genes": int(len(missing)),
                "missing_genes": ",".join(missing),
                "present_cells": None,
            }
        )
    if not perturbations:
        raise ValueError("No ScenicPlus regulon perturbations resolved to token dictionary entries.")
    return perturbations


def metadata_frame(dataset, label_column: str, donor_column: str) -> pd.DataFrame:
    columns = [
        "cell_id",
        donor_column,
        "donor_id",
        "SampleID",
        "sample_id",
        "PCW",
        "log2_PCW",
        "AgeGroup",
        "Lineage",
        "Cluster",
        "CellType",
        "Location",
    ]
    data = {}
    for column in dict.fromkeys(columns):
        if column in dataset.column_names:
            data[column] = dataset[column]
    meta = pd.DataFrame(data)
    meta["true_value"] = pd.to_numeric(pd.Series(dataset[label_column]), errors="coerce")
    if donor_column not in meta.columns:
        raise ValueError(f"Donor column {donor_column} missing from perturbation dataset.")
    pcw_values = pd.to_numeric(meta["PCW"], errors="coerce").to_numpy(dtype=float) if "PCW" in meta else None
    meta["pcw_bin"] = pcw_bin_column(pcw_values) if pcw_values is not None else None
    return meta


def aggregate_summary(per_cell: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (label, kind), group in per_cell.groupby(["perturbation_label", "perturbation_kind"], dropna=False):
        present = group["gene_present"].astype(bool)
        row = {
            "perturbation_label": label,
            "perturbation_kind": kind,
            "age_contribution_program": str(group["age_contribution_program"].iloc[0])
            if "age_contribution_program" in group
            else "",
            "n_cells": int(len(group)),
            "n_cells_with_perturbed_gene": int(present.sum()),
            "fraction_cells_with_perturbed_gene": float(present.mean()),
            "delta_pred_mean": float(group["delta_pred_value"].mean()),
            "delta_pred_median": float(group["delta_pred_value"].median()),
            "delta_pred_abs_median": float(group["delta_pred_value"].abs().median()),
            "embedding_l2_shift_mean": float(group["embedding_l2_shift"].mean()),
            "embedding_l2_shift_median": float(group["embedding_l2_shift"].median()),
            "embedding_cosine_original_perturbed_mean": float(group["embedding_cosine_original_perturbed"].mean()),
            "age_program_contribution_score_mean": float(group["age_program_contribution_score"].mean()),
            "age_program_contribution_score_median": float(group["age_program_contribution_score"].median()),
            "age_acceleration_score_mean": float(group["age_acceleration_score"].mean()),
            "age_acceleration_score_median": float(group["age_acceleration_score"].median()),
            "present_delta_pred_mean": float(group.loc[present, "delta_pred_value"].mean()) if present.any() else np.nan,
            "present_delta_pred_median": float(group.loc[present, "delta_pred_value"].median()) if present.any() else np.nan,
            "present_embedding_l2_shift_mean": float(group.loc[present, "embedding_l2_shift"].mean()) if present.any() else np.nan,
            "present_embedding_l2_shift_median": float(group.loc[present, "embedding_l2_shift"].median()) if present.any() else np.nan,
            "present_age_program_contribution_score_mean": float(
                group.loc[present, "age_program_contribution_score"].mean()
            )
            if present.any()
            else np.nan,
            "present_age_program_contribution_score_median": float(
                group.loc[present, "age_program_contribution_score"].median()
            )
            if present.any()
            else np.nan,
            "present_age_acceleration_score_mean": float(group.loc[present, "age_acceleration_score"].mean())
            if present.any()
            else np.nan,
            "present_age_acceleration_score_median": float(group.loc[present, "age_acceleration_score"].median())
            if present.any()
            else np.nan,
            "base_pred_mean": float(group["base_pred_value"].mean()),
            "perturbed_pred_mean": float(group["perturbed_pred_value"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("delta_pred_abs_median", ascending=False)


def contribution_program(perturbation: dict, adult_genes: set[str], fetal_genes: set[str], early_life_genes: set[str]) -> str:
    if not (adult_genes or fetal_genes or early_life_genes):
        return "unspecified"
    label = str(perturbation.get("perturbation_label", ""))
    if label == "adult_genes":
        return "adult"
    if label == "fetal_genes":
        return "fetal"
    if label == "early_life_genes":
        return "early_life_middle"
    genes = {
        str(gene).strip().upper()
        for gene in str(perturbation.get("target_genes", perturbation.get("input_gene", ""))).split(",")
        if str(gene).strip()
    }
    tf = str(perturbation.get("input_gene", "")).strip().upper()
    if tf == "TCF7L1" or label == "TCF7L1_regulon":
        return "fetal"
    if tf == "TCF7L1" or label.startswith("TCF7L1_scenicplus_"):
        return "fetal"
    if genes & adult_genes or tf in adult_genes:
        return "adult"
    if genes & fetal_genes or tf in fetal_genes:
        return "fetal"
    if genes & early_life_genes or tf in early_life_genes:
        return "early_life_middle"
    return "unspecified"


def age_contribution_score(
    base_pred: np.ndarray,
    perturbed_pred: np.ndarray,
    program: str,
    perturb_type: str,
    early_life_target: float,
) -> np.ndarray:
    delta = perturbed_pred - base_pred
    if perturb_type == "delete":
        expression_effect = -delta
    else:
        expression_effect = delta
    if program == "adult":
        return expression_effect.astype(float)
    if program == "fetal":
        return (-expression_effect).astype(float)
    if program == "early_life_middle":
        base_distance = np.abs(base_pred - early_life_target)
        perturbed_distance = np.abs(perturbed_pred - early_life_target)
        return (base_distance - perturbed_distance).astype(float)
    return np.full_like(delta, np.nan, dtype=float)


def age_acceleration_score(base_pred: np.ndarray, perturbed_pred: np.ndarray) -> np.ndarray:
    return (perturbed_pred - base_pred).astype(float)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune v10 epithelium-ileum log2 PCW regression and quantify donor-level age prediction shifts after perturbation."
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    parser.add_argument("--train-dataset", default=str(RUN_DIR / "tokenized" / "epithelium_ileum" / "train.dataset"))
    parser.add_argument("--test-dataset", default=str(RUN_DIR / "tokenized" / "epithelium_ileum" / "test.dataset"))
    parser.add_argument("--perturb-dataset-split", choices=["train", "test", "both"], default="test")
    parser.add_argument("--token-dictionary", default=str(DEFAULT_TOKEN_DICT))
    parser.add_argument("--gencode-gtf-gz", default=str(DEFAULT_GTF))
    parser.add_argument("--gene-mapping-tsv", default=str(DEFAULT_GENE_MAPPING_TSV))
    parser.add_argument("--label-column", default="log2_PCW")
    parser.add_argument("--donor-column", default="sample_id")
    parser.add_argument("--filter-json", default=None)
    parser.add_argument(
        "--train-filter-json",
        default=None,
        help="Optional JSON string/path for filtering regression-head training/evaluation; --filter-json filters perturbation cells.",
    )
    parser.add_argument("--adult-genes", default=",".join(DEFAULT_ADULT_GENES))
    parser.add_argument("--fetal-genes", default=",".join(DEFAULT_FETAL_GENES))
    parser.add_argument("--early-life-genes", default=",".join(DEFAULT_EARLY_LIFE_GENES))
    parser.add_argument("--genes", default=None, help="Optional extra comma-separated individual genes.")
    parser.add_argument("--gene-file", default=None)
    parser.add_argument("--include-individual-genes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--regulon-tsv", default=None)
    parser.add_argument("--signed-target-plan", default=None, help="Prepared signed expression-edit plan; replaces the usual gene/regulon panels")
    parser.add_argument("--signed-target-strengths", default="0.25,0.5,1")
    parser.add_argument("--save-regression-model", action="store_true")
    parser.add_argument("--only-regulons", action="store_true", help="Skip age gene panels; use --regulon-tsv only.")
    parser.add_argument("--regulon-tfs", default=",".join(DEFAULT_TFS))
    parser.add_argument("--regulon-include-tf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--regulon-max-targets", type=int, default=50)
    parser.add_argument("--regulon-confidence-levels", default=None)
    parser.add_argument("--regulon-signed-mode", choices=["all", "activation", "repression"], default="all")
    parser.add_argument("--include-scenicplus-regulons", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--only-scenicplus-regulons", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scenicplus-regulon-dir", default=str(DEFAULT_SCENICPLUS_REGULON_DIR))
    parser.add_argument("--scenicplus-regulon-tfs", default=",".join(DEFAULT_TFS))
    parser.add_argument(
        "--scenicplus-regulon-activities",
        default="all",
        help="Comma-separated ScenicPlus activity suffixes to use, e.g. '+_+,+_-,-_+,-_-'; default all available.",
    )
    parser.add_argument("--scenicplus-regulon-include-tf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--scenicplus-regulon-max-targets",
        type=int,
        default=0,
        help="Maximum ScenicPlus target genes per TF before optional TF inclusion; 0 uses all genes.",
    )
    parser.add_argument("--perturb-type", choices=["delete", "overexpress"], default="delete")
    parser.add_argument("--overexpress-value", type=float, default=None)
    parser.add_argument(
        "--early-life-target-log2-pcw",
        type=float,
        default=None,
        help="Middle-age target for early_life_genes contribution score; default is median true log2_PCW in perturbation cells.",
    )
    parser.add_argument("--max-cells", type=int, default=None, help="Optional cap on perturbation cells; default uses all selected cells.")
    parser.add_argument("--train-max-cells", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--predict-batch-size", type=int, default=16)
    parser.add_argument("--downstream-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--reference-baseline", required=True)
    parser.add_argument("--skip-training", action="store_true", help="Reuse the fitted regression model in --model-dir without fine-tuning")
    parser.add_argument("--save-per-cell", action="store_true")
    parser.add_argument("--umap-csv", default=str(RUN_DIR / "data" / "harmony_embedding" / "epithelium_ileum_umap_harmony.csv.gz"))
    parser.add_argument("--save-umap-cell-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default=str(RUN_DIR / "outputs" / "in_silico_age_perturbation" / DEFAULT_MODEL_KEY / f"epoch_{DEFAULT_EPOCH_LABEL}"))
    parser.add_argument("--output-prefix", default="v10_epithelium_ileum_log2_pcw_age_perturbation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.device == "cpu":
        raise RuntimeError("Mamba inference/fine-tuning requires CUDA in this environment; run on a GPU node.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_output_dir = output_dir / "regression_head_training"

    filters = load_json_arg(args.filter_json)
    train_filters = load_json_arg(args.train_filter_json)
    train_ds = load_from_disk(args.train_dataset)
    eval_ds = load_from_disk(args.test_dataset)
    train_ds = apply_filters(train_ds, train_filters, args.nproc)
    eval_ds = apply_filters(eval_ds, train_filters, args.nproc)
    train_ds = add_numeric_labels(train_ds, args.label_column)
    eval_ds = add_numeric_labels(eval_ds, args.label_column)
    train_ds = cap_dataset(train_ds, args.train_max_cells, args.seed)
    if len(train_ds) == 0:
        raise ValueError("No cells remain for regression-head training.")
    if len(eval_ds) == 0:
        raise ValueError("No cells remain for regression-head evaluation.")

    gene_to_token, _ = load_token_dictionary(Path(args.token_dictionary))
    perturbations = []
    if args.signed_target_plan:
        if args.regulon_tsv or args.include_scenicplus_regulons or args.only_scenicplus_regulons:
            raise ValueError("A signed-target plan is a separate experiment; do not combine with SCENIC/regulon flags")
        from signed_tf_targets import load_programs
        perturbations.extend(load_programs(args.signed_target_plan, args.signed_target_strengths))
        if perturbations[0]["signed_target_plan"]["perturb_type"] != args.perturb_type:
            raise ValueError("--perturb-type must match the signed target plan")
    if not args.signed_target_plan and not args.only_scenicplus_regulons and not args.only_regulons:
        perturbations.extend(build_gene_set_perturbations(args, gene_to_token))
    if args.regulon_tsv and not args.only_scenicplus_regulons:
        perturbations.extend(load_regulon_perturbations(args, gene_to_token))
    if args.include_scenicplus_regulons or args.only_scenicplus_regulons:
        perturbations.extend(load_scenicplus_regulon_perturbations(args, gene_to_token))
    if not perturbations:
        raise ValueError("No perturbations were resolved.")
    pd.DataFrame(perturbations).to_csv(output_dir / f"{args.output_prefix}_resolved_perturbations.csv", index=False)

    if not args.skip_training:
        raise ValueError('Rollout perturbations must reuse a saved model with --skip-training')
    from baseline_contract import strict_age_model, verify_age
    model = strict_age_model(args.model_dir)
    max_length = int(model.config.max_position_embeddings)
    collator = RegressionCollator(pad_token_id=int(model.config.pad_token_id), max_length=max_length)
    trainer = Trainer(
        model=model,
        args=build_training_args(args, train_output_dir),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_regression_metrics,
    )
    if args.skip_training:
        saved_config = json.loads((Path(args.model_dir)/'config.json').read_text())
        if saved_config.get('problem_type') != 'regression' or len(saved_config.get('id2label', {})) != 1:
            raise ValueError('--skip-training requires a previously fitted single-output regression model')
    train_result = None  # Fitting belongs exclusively to the saved-model stage.
    eval_metrics = trainer.evaluate()
    if args.save_regression_model:
        raise ValueError('Perturbation workers must not save/refit a separate age model')
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    perturb_ds = split_dataset(args.train_dataset, args.test_dataset, args.perturb_dataset_split)
    perturb_ds = apply_filters(perturb_ds, filters, args.nproc)
    perturb_ds = add_numeric_labels(perturb_ds, args.label_column)
    perturb_ds = cap_dataset(perturb_ds, args.max_cells, args.seed)
    if len(perturb_ds) == 0:
        raise ValueError("No cells remain for perturbation.")

    meta = metadata_frame(perturb_ds, args.label_column, args.donor_column)
    early_life_target = (
        float(args.early_life_target_log2_pcw)
        if args.early_life_target_log2_pcw is not None
        else float(pd.to_numeric(meta["true_value"], errors="coerce").median())
    )
    base_preds, base_emb = predict_and_embed_dataset(model, perturb_ds, collator, args.predict_batch_size, device)
    verify_age(args.reference_baseline, perturb_ds['cell_id'], base_preds, base_emb.numpy())
    cell_ids = [str(value) for value in perturb_ds["cell_id"]] if "cell_id" in perturb_ds.column_names else [str(i) for i in range(len(perturb_ds))]
    umap_coords = load_umap_coordinates(Path(args.umap_csv), cell_ids)
    adult_gene_set = {gene.upper() for gene in parse_csv_list(args.adult_genes)}
    fetal_gene_set = {gene.upper() for gene in parse_csv_list(args.fetal_genes)}
    early_life_gene_set = {gene.upper() for gene in parse_csv_list(args.early_life_genes)}

    per_cell_all = []
    perturbation_rows = []
    for perturbation in perturbations:
        tokens = [int(token) for token in perturbation.get("tokens", [perturbation["token"]])]
        token_set = set(tokens)
        pert_preds = []
        pert_embs = []
        present_mask = []
        edit_audits = []
        for _, _, rows in iter_batches(perturb_ds, args.predict_batch_size):
            present_mask.extend([bool({int(x) for x in row["input_ids"]} & token_set) for row in rows])
            if perturbation.get("signed_target_plan") is not None:
                from signed_tf_targets import edit_signed_rows
                perturbed_rows, audits = edit_signed_rows(rows, perturbation["signed_target_plan"],
                                                         perturbation["signed_strength"])
                edit_audits.extend(audits)
            else:
                perturbed_rows = perturb_rows(rows, tokens, args.perturb_type, args.overexpress_value)
            batch_preds, batch_embs = predict_and_embed_rows(model, perturbed_rows, collator, device)
            pert_preds.append(batch_preds)
            pert_embs.append(batch_embs)
        pert_preds_arr = np.concatenate(pert_preds, axis=0)
        pert_emb = torch.cat(pert_embs, dim=0)
        embedding_delta = pert_emb - base_emb
        embedding_l2_shift = torch.linalg.vector_norm(embedding_delta, dim=1).numpy()
        embedding_cosine = F.cosine_similarity(base_emb, pert_emb, dim=1).numpy()
        present = np.asarray(present_mask, dtype=bool)
        program = perturbation.get("age_contribution_program") or contribution_program(perturbation, adult_gene_set, fetal_gene_set, early_life_gene_set)
        contribution = age_contribution_score(
            base_preds,
            pert_preds_arr,
            program,
            args.perturb_type,
            early_life_target,
        )
        acceleration = age_acceleration_score(base_preds, pert_preds_arr)

        cell_df = meta.copy()
        cell_df["perturbation_label"] = perturbation["perturbation_label"]
        cell_df["perturbation_kind"] = perturbation.get("perturbation_kind", "single_gene")
        cell_df["age_contribution_program"] = program
        cell_df["early_life_target_log2_pcw"] = early_life_target
        cell_df["target_genes"] = perturbation.get("target_genes", perturbation.get("input_gene"))
        cell_df["n_tokens"] = int(perturbation.get("n_tokens", len(tokens)))
        cell_df["perturb_type"] = args.perturb_type
        cell_df["gene_present"] = present
        cell_df["base_pred_value"] = base_preds
        cell_df["perturbed_pred_value"] = pert_preds_arr
        cell_df["delta_pred_value"] = pert_preds_arr - base_preds
        cell_df["embedding_l2_shift"] = embedding_l2_shift
        cell_df["embedding_cosine_original_perturbed"] = embedding_cosine
        cell_df["age_program_contribution_score"] = contribution
        cell_df["age_acceleration_score"] = acceleration
        cell_df["edit_mode"] = perturbation.get("edit_mode", args.perturb_type)
        if edit_audits:
            for column in edit_audits[0]:
                cell_df[column] = [audit[column] for audit in edit_audits]
        per_cell_all.append(cell_df)

        perturbation_rows.append(
            {
                **perturbation,
                "perturb_type": args.perturb_type,
                "age_contribution_program": program,
                "early_life_target_log2_pcw": early_life_target,
                "n_cells": int(len(cell_df)),
                "n_cells_with_perturbed_gene": int(present.sum()),
                "fraction_cells_with_perturbed_gene": float(present.mean()),
                "delta_pred_mean": float(cell_df["delta_pred_value"].mean()),
                "delta_pred_median": float(cell_df["delta_pred_value"].median()),
                "delta_pred_abs_median": float(cell_df["delta_pred_value"].abs().median()),
                "embedding_l2_shift_mean": float(np.mean(embedding_l2_shift)),
                "embedding_l2_shift_median": float(np.median(embedding_l2_shift)),
                "embedding_cosine_original_perturbed_mean": float(np.mean(embedding_cosine)),
                "age_program_contribution_score_mean": float(np.nanmean(contribution)),
                "age_program_contribution_score_median": float(np.nanmedian(contribution)),
                "age_acceleration_score_mean": float(np.nanmean(acceleration)),
                "age_acceleration_score_median": float(np.nanmedian(acceleration)),
                "present_delta_pred_mean": float(cell_df.loc[present, "delta_pred_value"].mean()) if present.any() else np.nan,
                "present_delta_pred_median": float(cell_df.loc[present, "delta_pred_value"].median()) if present.any() else np.nan,
                "present_embedding_l2_shift_mean": float(np.mean(embedding_l2_shift[present])) if present.any() else np.nan,
                "present_embedding_l2_shift_median": float(np.median(embedding_l2_shift[present])) if present.any() else np.nan,
                "present_age_program_contribution_score_mean": float(np.nanmean(contribution[present])) if present.any() else np.nan,
                "present_age_program_contribution_score_median": float(np.nanmedian(contribution[present])) if present.any() else np.nan,
                "present_age_acceleration_score_mean": float(np.nanmean(acceleration[present])) if present.any() else np.nan,
                "present_age_acceleration_score_median": float(np.nanmedian(acceleration[present])) if present.any() else np.nan,
            }
        )
        print(
            f"{perturbation['perturbation_label']}: tokens={len(tokens)} present={present.sum()}/{len(present)} "
            f"median_delta={np.nanmedian(pert_preds_arr - base_preds):.4f}"
        )

    per_cell_df = pd.concat(per_cell_all, ignore_index=True)
    donor_df = donor_summary(per_cell_df, args.donor_column)
    summary_df = aggregate_summary(per_cell_df)
    perturbation_df = pd.DataFrame(perturbation_rows)

    donor_path = output_dir / f"{args.output_prefix}_{args.perturb_type}_donor_level_prediction_deltas.csv"
    summary_path = output_dir / f"{args.output_prefix}_{args.perturb_type}_summary.csv"
    perturbation_path = output_dir / f"{args.output_prefix}_{args.perturb_type}_perturbation_metrics.csv"
    donor_df.to_csv(donor_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    perturbation_df.to_csv(perturbation_path, index=False)
    if args.save_per_cell:
        write_rows(output_dir / f"{args.output_prefix}_{args.perturb_type}_per_cell_prediction_deltas.csv", per_cell_df.to_dict("records"))
    umap_metrics_path = None
    if args.save_umap_cell_metrics and umap_coords is not None:
        umap_cols = [
            "cell_id",
            "umap_1",
            "umap_2",
        ]
        metric_cols = [
            "perturbation_label",
            "perturbation_kind",
            "age_contribution_program",
            "perturb_type",
            "gene_present",
            "embedding_l2_shift",
            "embedding_cosine_original_perturbed",
            "base_pred_value",
            "perturbed_pred_value",
            "delta_pred_value",
            "age_program_contribution_score",
            "age_acceleration_score",
            args.donor_column,
            "PCW",
            "log2_PCW",
            "AgeGroup",
            "Lineage",
            "Cluster",
            "Location",
        ]
        umap_base = umap_coords.loc[:, umap_cols].drop_duplicates("cell_id")
        umap_df = per_cell_df.loc[:, [col for col in metric_cols if col in per_cell_df.columns]].copy()
        if "cell_id" not in umap_df.columns and "cell_id" in per_cell_df.columns:
            umap_df.insert(0, "cell_id", per_cell_df["cell_id"])
        umap_df = umap_df.merge(umap_base, on="cell_id", how="left")
        first_cols = ["cell_id", "umap_1", "umap_2"]
        ordered_cols = first_cols + [col for col in umap_df.columns if col not in first_cols]
        umap_metrics_path = output_dir / f"{args.output_prefix}_{args.perturb_type}_umap_cell_metrics.csv.gz"
        umap_df.loc[:, ordered_cols].to_csv(umap_metrics_path, index=False)

    manifest = {
        "reference_baseline": args.reference_baseline,
        "baseline_verified": True,
        "inference_precision": "float32; autocast off; TF32 off",
        "signed_target_plan": str(Path(args.signed_target_plan).resolve()) if args.signed_target_plan else None,
        "signed_target_strengths": args.signed_target_strengths if args.signed_target_plan else None,
        "trained_age_model": str(output_dir / "trained_age_model") if args.save_regression_model else None,
        "donor_aggregation": "biological_donor_v2",
        "model_dir": str(Path(args.model_dir).resolve()),
        "label_column": args.label_column,
        "donor_column": args.donor_column,
        "perturb_dataset_split": args.perturb_dataset_split,
        "n_train_cells": len(train_ds),
        "n_eval_cells": len(eval_ds),
        "n_perturb_cells": len(perturb_ds),
        "filters": filters,
        "train_filters": train_filters,
        "regulon_tsv": str(Path(args.regulon_tsv).resolve()) if args.regulon_tsv else None,
        "regulon_tfs": parse_csv_list(args.regulon_tfs),
        "regulon_include_tf": args.regulon_include_tf,
        "regulon_max_targets": args.regulon_max_targets,
        "only_regulons": args.only_regulons,
        "include_scenicplus_regulons": bool(args.include_scenicplus_regulons),
        "only_scenicplus_regulons": bool(args.only_scenicplus_regulons),
        "scenicplus_regulon_dir": str(Path(args.scenicplus_regulon_dir).resolve()),
        "scenicplus_regulon_tfs": parse_csv_list(args.scenicplus_regulon_tfs),
        "scenicplus_regulon_activities": parse_csv_list(args.scenicplus_regulon_activities),
        "scenicplus_regulon_include_tf": bool(args.scenicplus_regulon_include_tf),
        "scenicplus_regulon_max_targets": int(args.scenicplus_regulon_max_targets),
        "train_loss": float(train_result.training_loss) if train_result is not None else None,
        "training_skipped": args.skip_training,
        "eval_metrics": {key: float(value) for key, value in eval_metrics.items() if isinstance(value, (int, float, np.floating))},
        "resolved_perturbations_csv": str(output_dir / f"{args.output_prefix}_resolved_perturbations.csv"),
        "summary_csv": str(summary_path),
        "perturbation_metrics_csv": str(perturbation_path),
        "donor_level_prediction_deltas_csv": str(donor_path),
        "umap_csv": str(Path(args.umap_csv).resolve()) if Path(args.umap_csv).exists() else None,
        "umap_cell_metrics_csv": str(umap_metrics_path) if umap_metrics_path else None,
        "notes": [
            ("The fitted regression model is reused from --model-dir without training." if args.skip_training else
             "The regression head is fine-tuned from the selected group checkpoint because downstream comparison jobs did not save the trained regression head."),
            "Predictions and deltas are in the label space specified by --label-column, default log2_PCW.",
            "Donor-level rows are grouped by --donor-column and available age metadata, so early-life, fetal, and adult responses can be compared across PCW/AgeGroup.",
            "TF perturbations use the supplied TF-target table via --regulon-tsv and --regulon-tfs; compartment tables deduplicate targets across regions within each full signed eRegulon_name, preserving distinct sign combinations.",
            "Optional ScenicPlus TF-target perturbations use regulon_target_genes_and_regions.tsv Gene values from --scenicplus-regulon-dir.",
            "embedding_l2_shift and embedding_cosine_original_perturbed compare pooled regression-model embeddings before and after perturbation.",
            "umap_cell_metrics_csv joins original Seurat umap.harmony coordinates to cell-level perturbation metrics for plotting.",
        ],
    }
    (output_dir / f"{args.output_prefix}_{args.perturb_type}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {donor_path}")


if __name__ == "__main__":
    main()
