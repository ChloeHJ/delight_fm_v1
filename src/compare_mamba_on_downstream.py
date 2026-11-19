#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import disable_caching, load_from_disk
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, KFold
from transformers import Trainer, TrainingArguments, set_seed

from geneformer_mamba_model import GeneformerMambaForSequenceClassification


DCM_TRAIN_IDS = [
    "1447", "1600", "1462", "1558", "1300", "1508", "1358", "1678", "1561",
    "1304", "1610", "1430", "1472", "1707", "1726", "1504", "1425", "1617",
    "1631", "1735", "1582", "1722", "1622", "1630", "1290", "1479", "1371",
    "1549", "1515",
]
DCM_EVAL_IDS = ["1422", "1510", "1539", "1606", "1702"]


def parse_model_specs(raw_specs):
    parsed = []
    for spec in raw_specs:
        if "=" not in spec:
            raise ValueError(f"--model must look like name=/path/to/checkpoint, got: {spec}")
        name, path = spec.split("=", 1)
        parsed.append((name.strip(), path.strip()))
    return parsed


def parse_optional_int(raw):
    if raw is None:
        return None
    if str(raw).lower() in {"none", "all", "0", "null"}:
        return None
    return int(raw)


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


def filter_dataset(dataset, label_column, filter_json=None, split_column=None, split_ids=None):
    if label_column not in dataset.column_names:
        raise ValueError(
            f"Column '{label_column}' does not exist in this tokenized dataset. "
            f"Available columns: {dataset.column_names}"
        )
    filters = json.loads(filter_json) if filter_json else {}
    split_set = {str(x) for x in split_ids} if split_ids else None

    def keep(row):
        if normalize_missing(row.get(label_column)) is None:
            return False
        for key, allowed in filters.items():
            allowed_set = {str(x) for x in allowed}
            if str(row.get(key)) not in allowed_set:
                return False
        if split_column and split_set is not None and str(row.get(split_column)) not in split_set:
            return False
        return True

    return dataset.filter(keep)


def cap_per_class(dataset, label_name, max_per_class, seed):
    if max_per_class is None:
        return dataset
    df = pd.DataFrame({"idx": np.arange(len(dataset)), "label": dataset[label_name]})
    keep_idx = []
    for _, group in df.groupby("label", sort=True):
        sampled = group.sample(n=min(len(group), max_per_class), random_state=seed)
        keep_idx.extend(sampled["idx"].tolist())
    return dataset.select(sorted(keep_idx))


def maybe_cap_dataset(dataset, label_name, max_per_class, seed, regression=False):
    if regression:
        if max_per_class is not None:
            print(
                f"Skipping per-label capping for regression task on '{label_name}'. "
                "Use all numeric labels."
            )
        return dataset
    return cap_per_class(dataset, label_name, max_per_class, seed)


def maybe_cap_rows(dataset, max_rows, seed, name):
    if max_rows is None or len(dataset) <= max_rows:
        return dataset
    rng = np.random.default_rng(seed)
    keep_idx = np.sort(rng.choice(len(dataset), size=max_rows, replace=False))
    print(f"Capping {name} rows from {len(dataset)} to {max_rows}.")
    return dataset.select(keep_idx.tolist())


def add_categorical_labels(train_ds, eval_ds, label_column):
    train_values = [normalize_missing(x) for x in train_ds[label_column]]
    label_names = sorted({x for x in train_values if x is not None})
    label_to_id = {label: idx for idx, label in enumerate(label_names)}

    def encode(row):
        value = normalize_missing(row.get(label_column))
        row["labels"] = label_to_id.get(value, -1)
        return row

    train_ds = train_ds.map(encode).filter(lambda row: row["labels"] >= 0)
    eval_ds = eval_ds.map(encode).filter(lambda row: row["labels"] >= 0)
    return train_ds, eval_ds, label_names


def add_regression_labels(train_ds, eval_ds, label_column):
    def encode_batch(batch):
        values = pd.to_numeric(pd.Series(batch[label_column]), errors="coerce")
        batch["labels"] = [None if pd.isna(value) else float(value) for value in values]
        return batch

    train_ds = train_ds.map(encode_batch, batched=True)
    eval_ds = eval_ds.map(encode_batch, batched=True)
    return train_ds, eval_ds


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def add_sigmoid_pcw_regression_labels(
    train_ds,
    eval_ds,
    source_column,
    output_column,
    offset,
    amplitude,
    slope,
    midpoint_log2_pcw,
):
    def encode_batch(batch):
        pcw = pd.to_numeric(pd.Series(batch[source_column]), errors="coerce")
        log2_pcw = np.log2(pcw.astype(float).where(pcw > 0))
        transformed = offset + amplitude * sigmoid(slope * (log2_pcw - midpoint_log2_pcw))
        batch[output_column] = [None if pd.isna(value) else float(value) for value in transformed]
        batch["labels"] = batch[output_column]
        return batch

    train_ds = train_ds.map(encode_batch, batched=True)
    eval_ds = eval_ds.map(encode_batch, batched=True)
    return train_ds, eval_ds


def audit_regression_label_column(dataset, label_column, dataset_name, sample_size=10):
    if label_column not in dataset.column_names:
        raise ValueError(
            f"Column '{label_column}' does not exist in {dataset_name}. "
            f"Available columns: {dataset.column_names}"
        )
    raw_series = pd.Series(dataset[label_column], dtype="object")
    numeric_series = pd.to_numeric(raw_series, errors="coerce")
    valid_mask = numeric_series.notna()
    invalid_examples = raw_series[~valid_mask].drop_duplicates().head(sample_size).tolist()
    if not valid_mask.any():
        raise ValueError(f"Regression label column '{label_column}' in {dataset_name} has no numeric values.")
    if (~valid_mask).any():
        raise ValueError(
            f"Regression label column '{label_column}' in {dataset_name} contains "
            f"{int((~valid_mask).sum())} non-numeric values. Examples: {invalid_examples}"
        )
    valid_values = numeric_series[valid_mask].astype(float)
    unique_preview = sorted(valid_values.drop_duplicates().head(sample_size).tolist())
    print(
        f"{dataset_name} {label_column}: n={len(valid_values)}, "
        f"min={valid_values.min():.6f}, max={valid_values.max():.6f}, "
        f"mean={valid_values.mean():.6f}, n_unique={valid_values.nunique()}"
    )
    print(f"{dataset_name} {label_column} sample unique values: {unique_preview}")


class GeneformerPadCollator:
    def __init__(self, pad_token_id=0, max_length=2048, regression=False):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.regression = regression

    def __call__(self, features):
        lengths = [min(len(f["input_ids"]), self.max_length) for f in features]
        batch_len = max(lengths)
        input_ids = []
        attention_mask = []
        expression_values = []
        labels = []
        for feature, length in zip(features, lengths):
            ids = list(feature["input_ids"][:length])
            values = list(feature.get("expression_values", []))[:length]
            values = values + [0.0] * max(0, length - len(values))
            pad_len = batch_len - length
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * length + [0] * pad_len)
            expression_values.append(values + [0.0] * pad_len)
            labels.append(float(feature["labels"]) if self.regression else int(feature["labels"]))
        label_dtype = torch.float if self.regression else torch.long
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "expression_values": torch.tensor(expression_values, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=label_dtype),
        }


def build_training_args(args, output_dir):
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
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    if args.deepspeed_config:
        kwargs["deepspeed"] = args.deepspeed_config
    return TrainingArguments(**kwargs)


def compute_classification_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "weighted_precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "weighted_recall": recall_score(labels, preds, average="weighted", zero_division=0),
    }


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


def prediction_metadata(dataset, columns: list[str]) -> pd.DataFrame:
    data = {}
    for column in columns:
        if column in dataset.column_names:
            data[column] = dataset[column]
    return pd.DataFrame(data)


def donor_level_regression_summary(pred_df: pd.DataFrame, donor_column: str) -> pd.DataFrame:
    if donor_column not in pred_df.columns:
        return pd.DataFrame()
    group_cols = [donor_column]
    for optional in ("batch", "PCW", "age", "log2_PCW", "log2_PBW_ADJ"):
        if optional in pred_df.columns and optional not in group_cols:
            group_cols.append(optional)
    rows = []
    for keys, group in pred_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update({
            "n_cells": len(group),
            "true_mean": group["true_value"].mean(),
            "true_median": group["true_value"].median(),
            "pred_mean": group["pred_value"].mean(),
            "pred_median": group["pred_value"].median(),
            "residual_mean": group["residual"].mean(),
            "residual_median": group["residual"].median(),
            "residual_abs_median": group["residual"].abs().median(),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def classification_diagnostics(true_ids, pred_ids, label_names):
    label_ids = list(range(len(label_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        true_ids, pred_ids, labels=label_ids, zero_division=0
    )
    cm = confusion_matrix(true_ids, pred_ids, labels=label_ids)
    total = cm.sum()
    predicted = cm.sum(axis=0)
    per_class_df = pd.DataFrame({
        "class": label_names,
        "support": support.astype(int),
        "predicted": predicted.astype(int),
        "prediction_fraction": np.divide(predicted, total, out=np.zeros_like(predicted, dtype=float), where=total != 0),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

    off_diag = cm.astype(float)
    np.fill_diagonal(off_diag, 0)
    most_confused_pair = ""
    most_confused_count = 0.0
    if off_diag.size and off_diag.max() > 0:
        i, j = np.unravel_index(np.argmax(off_diag), off_diag.shape)
        most_confused_pair = f"{label_names[i]}->{label_names[j]}"
        most_confused_count = float(off_diag[i, j])

    aggregate = {
        "eval_macro_precision": precision_score(true_ids, pred_ids, average="macro", zero_division=0),
        "eval_macro_recall": recall_score(true_ids, pred_ids, average="macro", zero_division=0),
        "eval_balanced_accuracy": balanced_accuracy_score(true_ids, pred_ids),
        "eval_weighted_precision": precision_score(true_ids, pred_ids, average="weighted", zero_division=0),
        "eval_weighted_recall": recall_score(true_ids, pred_ids, average="weighted", zero_division=0),
        "worst_class_recall": float(per_class_df["recall"].min()),
        "worst_class_recall_label": str(per_class_df.loc[per_class_df["recall"].idxmin(), "class"]),
        "worst_class_precision": float(per_class_df["precision"].min()),
        "worst_class_precision_label": str(per_class_df.loc[per_class_df["precision"].idxmin(), "class"]),
        "dominant_predicted_class": str(label_names[int(np.argmax(predicted))]) if label_names else "",
        "dominant_prediction_fraction": float(predicted.max() / total) if total else 0.0,
        "most_confused_pair": most_confused_pair,
        "most_confused_count": most_confused_count,
    }
    return aggregate, per_class_df, cm


def save_predictions_and_confusion(trainer, eval_ds, label_names, run_output_dir):
    prediction_output = trainer.predict(eval_ds)
    predictions = prediction_output.predictions
    labels = prediction_output.label_ids
    pred_ids = np.argmax(predictions, axis=-1).astype(int)
    true_ids = labels.astype(int)
    id_to_label = {idx: label for idx, label in enumerate(label_names)}

    run_output_dir = Path(run_output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "true_id": true_ids,
        "pred_id": pred_ids,
        "true_label": [id_to_label.get(i, str(i)) for i in true_ids],
        "pred_label": [id_to_label.get(i, str(i)) for i in pred_ids],
    }).to_csv(run_output_dir / "eval_predictions.csv", index=False)

    aggregate, per_class_df, cm = classification_diagnostics(true_ids, pred_ids, label_names)
    pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(run_output_dir / "confusion_matrix_counts.csv")
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    pd.DataFrame(cm_norm, index=label_names, columns=label_names).to_csv(
        run_output_dir / "confusion_matrix_row_normalized.csv"
    )
    per_class_df.to_csv(run_output_dir / "per_class_metrics.csv", index=False)
    return aggregate


def save_regression_predictions(trainer, eval_ds, run_output_dir, metadata_columns=None, donor_column="sample_id"):
    prediction_output = trainer.predict(eval_ds)
    predictions = np.squeeze(prediction_output.predictions).astype(float)
    labels = prediction_output.label_ids.astype(float)
    run_output_dir = Path(run_output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({
        "true_value": labels,
        "pred_value": predictions,
        "residual": predictions - labels,
    })
    if metadata_columns:
        meta = prediction_metadata(eval_ds, metadata_columns)
        pred_df = pd.concat([meta.reset_index(drop=True), pred_df], axis=1)
    pred_df.to_csv(run_output_dir / "eval_predictions.csv", index=False)
    donor_df = donor_level_regression_summary(pred_df, donor_column)
    if not donor_df.empty:
        donor_df.to_csv(run_output_dir / "donor_level_predictions.csv", index=False)

    residuals = predictions - labels
    metrics = {
        "residual_mean": float(np.mean(residuals)),
        "residual_median": float(np.median(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_abs_median": float(np.median(np.abs(residuals))),
    }
    if not donor_df.empty and len(donor_df) > 1:
        metrics.update({
            "donor_level_rmse": float(np.sqrt(mean_squared_error(donor_df["true_mean"], donor_df["pred_mean"]))),
            "donor_level_mae": float(mean_absolute_error(donor_df["true_mean"], donor_df["pred_mean"])),
            "donor_level_r2": float(r2_score(donor_df["true_mean"], donor_df["pred_mean"])),
            "donor_level_pearson": float(np.corrcoef(donor_df["true_mean"], donor_df["pred_mean"])[0, 1]),
        })
    return metrics


def run_regression_cv(args, train_ds, model_name: str, model_path: str, output_dir: Path, collator):
    if args.cv_folds <= 1:
        return {}
    if args.cv_group_column and args.cv_group_column in train_ds.column_names:
        groups = np.asarray(train_ds[args.cv_group_column]).astype(str)
        splitter = GroupKFold(n_splits=args.cv_folds)
        split_iter = splitter.split(np.arange(len(train_ds)), groups=groups)
        split_kind = f"grouped by {args.cv_group_column}"
    else:
        splitter = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(np.arange(len(train_ds)))
        split_kind = "cell-level shuffled"

    cv_root = output_dir / f"{args.output_prefix}_{model_name}_cv"
    cv_root.mkdir(parents=True, exist_ok=True)
    metadata_columns = parse_csv_list(args.prediction_metadata_columns)
    fold_rows = []
    all_predictions = []
    for fold_idx, (train_idx, val_idx) in enumerate(split_iter, start=1):
        fold_dir = cv_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_train = train_ds.select(sorted(train_idx.tolist()))
        fold_val = train_ds.select(sorted(val_idx.tolist()))
        print(f"CV fold {fold_idx}/{args.cv_folds} ({split_kind}): train={len(fold_train)}, val={len(fold_val)}")
        model = GeneformerMambaForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1,
            problem_type="regression",
            ignore_mismatched_sizes=True,
        )
        trainer = Trainer(
            model=model,
            args=build_training_args(args, fold_dir),
            train_dataset=fold_train,
            eval_dataset=fold_val,
            data_collator=collator,
            compute_metrics=compute_regression_metrics,
        )
        train_result = trainer.train()
        eval_metrics = trainer.evaluate()
        fold_dir.mkdir(parents=True, exist_ok=True)
        extra = save_regression_predictions(
            trainer,
            fold_val,
            fold_dir,
            metadata_columns=metadata_columns,
            donor_column=args.donor_column,
        )
        pred_path = fold_dir / "eval_predictions.csv"
        fold_pred = pd.read_csv(pred_path)
        fold_pred["fold"] = fold_idx
        all_predictions.append(fold_pred)
        fold_row = {
            "fold": fold_idx,
            "split_kind": split_kind,
            "train_cells": len(fold_train),
            "val_cells": len(fold_val),
            "train_loss": train_result.training_loss,
            "eval_loss": eval_metrics.get("eval_loss"),
            "eval_mae": eval_metrics.get("eval_mae"),
            "eval_rmse": eval_metrics.get("eval_rmse"),
            "eval_r2": eval_metrics.get("eval_r2"),
            "eval_pearson": eval_metrics.get("eval_pearson"),
            **extra,
        }
        fold_rows.append(fold_row)
    cv_df = pd.DataFrame(fold_rows)
    cv_df.to_csv(cv_root / "cv_summary.csv", index=False)
    if all_predictions:
        cv_pred = pd.concat(all_predictions, ignore_index=True)
        cv_pred.to_csv(cv_root / "cv_eval_predictions.csv", index=False)
        donor_df = donor_level_regression_summary(cv_pred, args.donor_column)
        if not donor_df.empty:
            donor_df.to_csv(cv_root / "cv_donor_level_predictions.csv", index=False)
    metrics = {
        "cv_folds": args.cv_folds,
        "cv_mean_rmse": cv_df["eval_rmse"].mean(),
        "cv_mean_mae": cv_df["eval_mae"].mean(),
        "cv_mean_r2": cv_df["eval_r2"].mean(),
        "cv_mean_pearson": cv_df["eval_pearson"].mean(),
        "cv_summary_path": str(cv_root / "cv_summary.csv"),
    }
    return metrics


def write_model_comparison_stability(rows, output_dir, output_prefix):
    if not rows:
        return
    df = pd.DataFrame(rows)
    mode = str(df.iloc[0].get("mode", "classification"))
    if mode == "regression":
        best = df.loc[df["eval_rmse"].astype(float).idxmin()]
        stability = {
            "best_model_name": best["model_name"],
            "best_model_path": best["model_path"],
            "best_rmse": best.get("eval_rmse"),
            "best_mae": best.get("eval_mae"),
            "best_r2": best.get("eval_r2"),
            "best_pearson": best.get("eval_pearson"),
        }
    else:
        best = df.loc[df["eval_macro_f1"].astype(float).idxmax()]
        stability = {
            "best_model_name": best["model_name"],
            "best_model_path": best["model_path"],
            "best_macro_f1": best.get("eval_macro_f1"),
            "best_macro_precision": best.get("eval_macro_precision"),
            "best_macro_recall": best.get("eval_macro_recall"),
            "best_balanced_accuracy": best.get("eval_balanced_accuracy"),
            "best_accuracy": best.get("eval_accuracy"),
        }
    path = Path(output_dir) / f"{output_prefix}_stability_summary.csv"
    pd.DataFrame([stability]).to_csv(path, index=False)
    print(f"Wrote stability summary: {path}")


def load_dcm_datasets(args):
    full_ds = load_from_disk(args.dcm_dataset)
    filter_json = args.filter_json or json.dumps({
        "cell_type": ["Cardiomyocyte1", "Cardiomyocyte2", "Cardiomyocyte3"]
    })
    train_ds = filter_dataset(
        full_ds,
        args.label_column,
        filter_json=filter_json,
        split_column="individual",
        split_ids=DCM_TRAIN_IDS,
    )
    eval_ds = filter_dataset(
        full_ds,
        args.label_column,
        filter_json=filter_json,
        split_column="individual",
        split_ids=DCM_EVAL_IDS,
    )
    return train_ds, eval_ds


def load_task_datasets(args):
    if args.train_dataset or args.eval_dataset:
        if not args.train_dataset or not args.eval_dataset:
            raise ValueError("--train-dataset and --eval-dataset must be provided together.")
        return load_from_disk(args.train_dataset), load_from_disk(args.eval_dataset)
    return load_dcm_datasets(args)


def main():
    disable_caching()

    parser = argparse.ArgumentParser(description="Fine-tune/evaluate Mamba checkpoints on downstream tasks.")
    parser.add_argument("--task", default="dcm")
    parser.add_argument("--model", action="append", required=True, help="Repeatable. Format: name=/path/to/checkpoint.")
    parser.add_argument("--dcm-dataset", default=None)
    parser.add_argument("--train-dataset", default=None)
    parser.add_argument("--eval-dataset", default=None)
    parser.add_argument("--label-column", default="disease")
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--regression-label-transform", choices=["none", "pcw_sigmoid"], default="none")
    parser.add_argument("--pcw-transform-source-column", default="PCW")
    parser.add_argument("--pcw-transform-output-column", default="transformed_PCW")
    parser.add_argument("--pcw-transform-json", default=None)
    parser.add_argument("--pcw-transform-offset", type=float, default=-1.0)
    parser.add_argument("--pcw-transform-amplitude", type=float, default=2.0)
    parser.add_argument("--pcw-transform-slope", type=float, default=1.0)
    parser.add_argument("--pcw-transform-midpoint-log2", type=float, default=4.0)
    parser.add_argument("--filter-json", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="mamba_disease_compare")
    parser.add_argument("--max-ncells-per-class", default="2000")
    parser.add_argument("--max-train-cells", default=None)
    parser.add_argument("--max-eval-cells", default=None)
    parser.add_argument("--downstream-epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=8.04e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--deepspeed-config", default=None)
    parser.add_argument("--prediction-metadata-columns", default="sample_id,batch,PCW,age,log2_PCW,log2_PBW_ADJ,BroadCluster,JointCluster")
    parser.add_argument("--donor-column", default="sample_id")
    parser.add_argument("--cv-folds", type=int, default=0)
    parser.add_argument("--cv-group-column", default="sample_id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.train_dataset and not args.dcm_dataset:
        parser.error("Supply --train-dataset and --eval-dataset, or an explicit --dcm-dataset")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_ncells_per_class = parse_optional_int(args.max_ncells_per_class)
    model_specs = parse_model_specs(args.model)

    train_ds, eval_ds = load_task_datasets(args)
    if args.regression:
        if args.regression_label_transform == "pcw_sigmoid":
            transform = {}
            if args.pcw_transform_json:
                transform_path = Path(args.pcw_transform_json)
                transform = json.loads(transform_path.read_text())
                transform = transform.get("learned_pcw_scale", transform)
                print(f"Regression target source: sigmoid PCW transform from {transform_path}")
            offset = float(transform.get("offset", args.pcw_transform_offset))
            amplitude = float(transform.get("amplitude", args.pcw_transform_amplitude))
            slope = float(transform.get("slope", args.pcw_transform_slope))
            midpoint_log2_pcw = float(transform.get("midpoint_log2_pcw", args.pcw_transform_midpoint_log2))
            for dataset, name in [(train_ds, "train dataset"), (eval_ds, "eval dataset")]:
                if args.pcw_transform_source_column not in dataset.column_names:
                    raise ValueError(
                        f"Column '{args.pcw_transform_source_column}' does not exist in {name}. "
                        f"Available columns: {dataset.column_names}"
                    )
            print(
                "Regression target source: "
                f"{args.pcw_transform_output_column} = {offset:.6g} + {amplitude:.6g} * sigmoid("
                f"{slope:.6g} * (log2({args.pcw_transform_source_column}) - {midpoint_log2_pcw:.6g}))"
            )
            train_ds, eval_ds = add_sigmoid_pcw_regression_labels(
                train_ds,
                eval_ds,
                args.pcw_transform_source_column,
                args.pcw_transform_output_column,
                offset,
                amplitude,
                slope,
                midpoint_log2_pcw,
            )
            args.label_column = args.pcw_transform_output_column
        else:
            audit_regression_label_column(train_ds, args.label_column, "train dataset")
            audit_regression_label_column(eval_ds, args.label_column, "eval dataset")
            if args.label_column == "log2_PCW":
                print("Regression target source: using existing dataset column 'log2_PCW' (built as log2(PCW) when absent).")
            elif args.label_column == "PCW":
                print("Regression target source: using raw dataset column 'PCW' without log2 transformation.")
            train_ds, eval_ds = add_regression_labels(train_ds, eval_ds, args.label_column)
        label_names = ["continuous"]
    else:
        train_ds, eval_ds, label_names = add_categorical_labels(train_ds, eval_ds, args.label_column)
    train_ds = maybe_cap_dataset(train_ds, "labels", max_ncells_per_class, args.seed, regression=args.regression)
    eval_ds = maybe_cap_dataset(eval_ds, "labels", max_ncells_per_class, args.seed, regression=args.regression)
    train_ds = maybe_cap_rows(train_ds, parse_optional_int(args.max_train_cells), args.seed, "train dataset")
    eval_ds = maybe_cap_rows(eval_ds, parse_optional_int(args.max_eval_cells), args.seed, "eval dataset")

    print(f"Task: {args.task}")
    print(f"Label column: {args.label_column}")
    print(f"Mode: {'regression' if args.regression else 'classification'}")
    print(f"Labels ({len(label_names)}): {label_names}")
    print(f"Train cells after filtering/capping: {len(train_ds)}")
    print(f"Eval cells after filtering/capping: {len(eval_ds)}")
    print("Models:")
    for name, path in model_specs:
        print(f"  {name}: {path}")

    if args.dry_run:
        return

    rows = []
    collator = GeneformerPadCollator(max_length=args.max_length, regression=args.regression)
    metadata_columns = parse_csv_list(args.prediction_metadata_columns)
    for model_name, model_path in model_specs:
        run_output_dir = output_dir / f"{args.output_prefix}_{model_name}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Fine-tune/eval: {model_name} ===")
        cv_metrics = {}
        if args.regression and args.cv_folds > 1:
            cv_metrics = run_regression_cv(args, train_ds, model_name, model_path, output_dir, collator)

        model_kwargs = {
            "num_labels": 1 if args.regression else len(label_names),
            "ignore_mismatched_sizes": True,
        }
        if args.regression:
            model_kwargs["problem_type"] = "regression"
        else:
            model_kwargs["id2label"] = {i: label for i, label in enumerate(label_names)}
            model_kwargs["label2id"] = {label: i for i, label in enumerate(label_names)}
        model = GeneformerMambaForSequenceClassification.from_pretrained(model_path, **model_kwargs)
        trainer = Trainer(
            model=model,
            args=build_training_args(args, run_output_dir),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            compute_metrics=compute_regression_metrics if args.regression else compute_classification_metrics,
        )
        train_result = trainer.train()
        eval_metrics = trainer.evaluate()
        if args.regression:
            extra_metrics = save_regression_predictions(
                trainer,
                eval_ds,
                run_output_dir,
                metadata_columns=metadata_columns,
                donor_column=args.donor_column,
            )
        else:
            extra_metrics = save_predictions_and_confusion(trainer, eval_ds, label_names, run_output_dir)

        row = {
            "task": args.task,
            "label_column": args.label_column,
            "model_name": model_name,
            "model_path": model_path,
            "mode": "regression" if args.regression else "classification",
            "num_labels": len(label_names),
            "train_cells": len(train_ds),
            "eval_cells": len(eval_ds),
            "downstream_epochs": args.downstream_epochs,
            "train_loss": train_result.training_loss,
            "eval_loss": eval_metrics.get("eval_loss"),
            "eval_accuracy": eval_metrics.get("eval_accuracy"),
            "eval_macro_f1": eval_metrics.get("eval_macro_f1"),
            "eval_macro_precision": eval_metrics.get("eval_macro_precision"),
            "eval_macro_recall": eval_metrics.get("eval_macro_recall"),
            "eval_balanced_accuracy": eval_metrics.get("eval_balanced_accuracy"),
            "eval_weighted_f1": eval_metrics.get("eval_weighted_f1"),
            "eval_weighted_precision": eval_metrics.get("eval_weighted_precision"),
            "eval_weighted_recall": eval_metrics.get("eval_weighted_recall"),
            "eval_mae": eval_metrics.get("eval_mae"),
            "eval_mse": eval_metrics.get("eval_mse"),
            "eval_rmse": eval_metrics.get("eval_rmse"),
            "eval_r2": eval_metrics.get("eval_r2"),
            "eval_pearson": eval_metrics.get("eval_pearson"),
            "output_dir": str(run_output_dir),
            **cv_metrics,
            **extra_metrics,
        }
        rows.append(row)

        summary_csv = output_dir / f"{args.output_prefix}_summary.csv"
        with summary_csv.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote partial summary: {summary_csv}")

    summary_json = output_dir / f"{args.output_prefix}_summary.json"
    with summary_json.open("w") as fp:
        json.dump(rows, fp, indent=2)
    write_model_comparison_stability(rows, output_dir, args.output_prefix)
    print(f"\nDone: {output_dir / f'{args.output_prefix}_summary.csv'}")


if __name__ == "__main__":
    main()
