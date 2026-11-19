#!/usr/bin/env python
from __future__ import annotations

import csv
import datetime
import inspect
import json
import math
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from token_padding import GeneTokenPadding
from geneformer_mamba_model import GeneformerMambaConfig, GeneformerMambaForAtlasPretraining


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def env_float(name, default):
    return float(os.environ.get(name, str(default)))


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def build_training_args(kwargs):
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return TrainingArguments(**kwargs)


def write_log_history(log_history, csv_path: Path, json_path: Path) -> None:
    rows = [dict(row) for row in log_history]
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as fp:
        json.dump(rows, fp, indent=2)
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric_column(dataset, column: str) -> np.ndarray:
    if column not in dataset.column_names:
        return np.asarray([], dtype=np.float64)
    numeric = []
    for value in dataset[column]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numeric.append(number)
    return np.asarray(numeric, dtype=np.float64)


def compute_pcw_stats(dataset, pcw_column: str) -> dict:
    pcw = numeric_column(dataset, pcw_column)
    if pcw.size == 0:
        return {}
    if np.any(pcw <= 0):
        raise ValueError("PCW must be positive for log2(PCW)")
    log2_pcw = np.log2(pcw)
    return {
        "pcw_column": pcw_column,
        "pcw_min": float(np.min(pcw)),
        "pcw_median": float(np.median(pcw)),
        "pcw_max": float(np.max(pcw)),
        "pcw_log2_min": float(np.min(log2_pcw)),
        "pcw_log2_median": float(np.median(log2_pcw)),
        "pcw_log2_max": float(np.max(log2_pcw)),
        "n_examples": int(pcw.size),
    }


@dataclass
class AtlasMambaCollator:
    token_dictionary: dict
    mlm_probability: float = 0.15
    pcw_column: str = "PCW"

    def __post_init__(self):
        precollator = GeneTokenPadding(token_dictionary=self.token_dictionary)
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=precollator,
            mlm=True,
            mlm_probability=self.mlm_probability,
        )

    def __call__(self, features):
        mlm_features = [{"input_ids": feature["input_ids"]} for feature in features]
        batch = self.mlm_collator(mlm_features)

        seq_len = batch["input_ids"].shape[1]
        expression_values = []
        auxiliary_gene_mask = []
        for feature in features:
            values = list(feature.get("expression_values", []))[:seq_len]
            values = values + [0.0] * (seq_len - len(values))
            expression_values.append(values)
            mask_values = list(feature.get("auxiliary_gene_mask", []))[:seq_len]
            mask_values = mask_values + [0] * (seq_len - len(mask_values))
            auxiliary_gene_mask.append(mask_values)

        batch["expression_values"] = torch.tensor(expression_values, dtype=torch.float32)
        batch["auxiliary_gene_mask"] = torch.tensor(auxiliary_gene_mask, dtype=torch.long)
        batch["cluster_labels"] = torch.tensor([int(feature.get("Cluster_id", -1)) for feature in features], dtype=torch.long)
        batch["lineage_labels"] = torch.tensor([int(feature.get("Lineage_id", -1)) for feature in features], dtype=torch.long)
        batch["study_labels"] = torch.tensor([int(feature.get("study_id", -1)) for feature in features], dtype=torch.long)
        batch["orig_ident_labels"] = torch.tensor([int(feature.get("orig_ident_id", -1)) for feature in features], dtype=torch.long)
        batch["multiome_labels"] = torch.tensor([int(feature.get("multiome_id", -1)) for feature in features], dtype=torch.long)
        pcw_values = []
        for feature in features:
            try:
                pcw_values.append(float(feature.get(self.pcw_column, float("nan"))))
            except (TypeError, ValueError):
                pcw_values.append(float("nan"))
        batch["pcw_values"] = torch.tensor(pcw_values, dtype=torch.float32)
        return batch


def main():
    seed_val = env_int("GENEFORMER_SEED", 42)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    from paths import RUN as default_run
    run_root = Path(os.environ.get("ATLAS_MAMBA_RUN_ROOT", default_run))
    dataset_root = Path(os.environ.get("ATLAS_MAMBA_DATASET_ROOT", run_root / "tokenized"))
    train_dataset_path = Path(os.environ.get("GENEFORMER_DATASET_PATH", dataset_root / "train.dataset"))
    eval_dataset_path = Path(os.environ.get("GENEFORMER_EVAL_DATASET_PATH", dataset_root / "test.dataset"))
    token_dictionary_file = Path(os.environ["GENEFORMER_TOKEN_DICTIONARY"])
    label_maps_file = Path(os.environ.get("ATLAS_MAMBA_LABEL_MAPS", dataset_root / "label_maps.json"))

    with token_dictionary_file.open("rb") as fp:
        token_dictionary = pickle.load(fp)
    with label_maps_file.open() as fp:
        label_maps = json.load(fp)

    train_dataset = load_from_disk(str(train_dataset_path))
    eval_dataset = load_from_disk(str(eval_dataset_path)) if eval_dataset_path.exists() else None

    subset_size = env_int("GENEFORMER_SUBSET_SIZE", 0)
    eval_subset_size = env_int("GENEFORMER_EVAL_SUBSET_SIZE", 0)
    if subset_size > 0:
        train_dataset = train_dataset.select(range(min(subset_size, len(train_dataset))))
    if eval_dataset is not None and eval_subset_size > 0:
        eval_dataset = eval_dataset.select(range(min(eval_subset_size, len(eval_dataset))))

    max_input_size = env_int("GENEFORMER_MAX_INPUT_SIZE", 2048)
    num_layers = env_int("MAMBA_NUM_LAYERS", 4)
    hidden_size = env_int("MAMBA_HIDDEN_SIZE", 256)
    state_size = env_int("MAMBA_STATE_SIZE", 16)
    conv_kernel = env_int("MAMBA_CONV_KERNEL", 4)
    expand = env_int("MAMBA_EXPAND", 2)
    hidden_dropout_prob = env_float("GENEFORMER_HIDDEN_DROPOUT", 0.02)
    mlm_probability = env_float("GENEFORMER_MLM_PROBABILITY", 0.15)
    pcw_column = os.environ.get("GENEFORMER_PCW_COLUMN", "PCW")
    batch_size = env_int("GENEFORMER_BATCH_SIZE", 8)
    epochs = env_float("GENEFORMER_EPOCHS", 10.0)
    num_gpus = env_int("WORLD_SIZE", env_int("NUM_GPUS", 1))
    pcw_stats = compute_pcw_stats(train_dataset, pcw_column)
    pcw_plateau_log2_init = env_float(
        "GENEFORMER_PCW_PLATEAU_LOG2_INIT",
        pcw_stats.get("pcw_log2_median", 8.0),
    )
    steps_per_epoch = max(1, int(np.floor(len(train_dataset) / batch_size / max(num_gpus, 1))))
    total_steps = max(1, int(np.ceil(steps_per_epoch * epochs)))
    warmup_steps = env_int("GENEFORMER_WARMUP_STEPS", max(1, int(total_steps * env_float("GENEFORMER_WARMUP_RATIO", 0.02))))
    save_steps = env_int("GENEFORMER_SAVE_STEPS", max(1, int(steps_per_epoch * env_float("GENEFORMER_SAVE_FRACTION", 0.1))))
    eval_steps = env_int("GENEFORMER_EVAL_STEPS", save_steps)

    run_name = os.environ.get("GENEFORMER_RUN_NAME")
    if not run_name:
        stamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        run_name = f"{stamp}_atlas_value_mamba_L{num_layers}_d{hidden_size}"

    output_root = Path(os.environ.get("GENEFORMER_ROOTDIR", run_root / "pretrain_outputs"))
    training_output_dir = output_root / "models" / run_name
    logging_dir = output_root / "runs" / run_name
    metrics_dir = Path(os.environ.get("GENEFORMER_METRICS_DIR", output_root / "metrics" / run_name))

    config = GeneformerMambaConfig(
        vocab_size=len(token_dictionary),
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        state_size=state_size,
        conv_kernel=conv_kernel,
        expand=expand,
        max_position_embeddings=max_input_size,
        pad_token_id=token_dictionary.get("<pad>"),
        hidden_dropout_prob=hidden_dropout_prob,
        use_expression_values=env_bool("ATLAS_MAMBA_USE_EXPRESSION_VALUES", True),
        num_cluster_labels=len(label_maps.get("Cluster", {})),
        num_lineage_labels=len(label_maps.get("Lineage", {})),
        num_study_labels=len(label_maps.get("study", {})),
        num_orig_ident_labels=len(label_maps.get("orig.ident", {})),
        num_multiome_labels=len(label_maps.get("multiome", {})),
        cluster_auxiliary_loss_weight=env_float("ATLAS_MAMBA_CLUSTER_LOSS_WEIGHT", 0.05),
        lineage_auxiliary_loss_weight=env_float("ATLAS_MAMBA_LINEAGE_LOSS_WEIGHT", 0.02),
        study_adversarial_loss_weight=env_float("ATLAS_MAMBA_STUDY_ADV_LOSS_WEIGHT", 0.02),
        orig_ident_adversarial_loss_weight=env_float("ATLAS_MAMBA_ORIG_IDENT_ADV_LOSS_WEIGHT", 0.02),
        multiome_adversarial_loss_weight=env_float("ATLAS_MAMBA_MULTIOME_ADV_LOSS_WEIGHT", 0.02),
        pcw_auxiliary_loss_weight=env_float("GENEFORMER_PCW_LOSS_WEIGHT", 0.1 if pcw_stats else 0.0),
        pcw_rank_loss_weight=env_float("GENEFORMER_PCW_RANK_LOSS_WEIGHT", 0.02 if pcw_stats else 0.0),
        pcw_log2_min=pcw_stats.get("pcw_log2_min", 0.0),
        pcw_log2_max=pcw_stats.get("pcw_log2_max", 12.0),
        pcw_plateau_log2_init=pcw_plateau_log2_init,
        pcw_plateau_slope_init=env_float("GENEFORMER_PCW_PLATEAU_SLOPE_INIT", 1.0),
        pcw_huber_beta=env_float("GENEFORMER_PCW_HUBER_BETA", 0.5),
    )
    initial_model_path = os.environ.get("GENEFORMER_INITIAL_MODEL_PATH", "").strip()
    if initial_model_path:
        print(f"Initializing model weights from: {initial_model_path}")
        model = GeneformerMambaForAtlasPretraining.from_pretrained(
            initial_model_path,
            config=config,
        ).train()
        if env_bool("ATLAS_MAMBA_RESET_PCW_SCALE", False):
            model.pcw_scale = type(model.pcw_scale)(config)
            print("Reset PCW scale parameters after loading initial model.")
        if env_bool("ATLAS_MAMBA_RESET_PCW_HEAD", False):
            model.pcw_head = torch.nn.Sequential(
                torch.nn.LayerNorm(config.hidden_size),
                torch.nn.Dropout(config.hidden_dropout_prob),
                torch.nn.Linear(config.hidden_size, 1),
            )
            print("Reset PCW prediction head after loading initial model.")
    else:
        model = GeneformerMambaForAtlasPretraining(config).train()

    training_kwargs = {
        "learning_rate": env_float("GENEFORMER_LR", 3e-4),
        "do_train": True,
        "do_eval": eval_dataset is not None,
        "evaluation_strategy": "steps" if eval_dataset is not None else "no",
        "eval_steps": eval_steps,
        "group_by_length": True,
        "length_column_name": "length",
        "remove_unused_columns": False,
        "lr_scheduler_type": os.environ.get("GENEFORMER_LR_SCHEDULE", "linear"),
        "warmup_steps": warmup_steps,
        "weight_decay": env_float("GENEFORMER_WEIGHT_DECAY", 0.05),
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": env_int("GENEFORMER_EVAL_BATCH_SIZE", batch_size),
        "num_train_epochs": epochs,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "logging_strategy": "steps",
        "logging_steps": env_int("GENEFORMER_LOGGING_STEPS", 200),
        "output_dir": str(training_output_dir),
        "logging_dir": str(logging_dir),
        "bf16": env_bool("GENEFORMER_BF16", True),
        "report_to": [],
    }
    deepspeed_config = os.environ.get("GENEFORMER_DEEPSPEED_CONFIG", "").strip()
    if deepspeed_config:
        training_kwargs["deepspeed"] = deepspeed_config
    training_args = build_training_args(training_kwargs)

    print(f"Run name: {run_name}")
    print(f"Train dataset: {train_dataset_path} ({len(train_dataset)} cells)")
    print(f"Eval dataset: {eval_dataset_path if eval_dataset is not None else 'disabled'} ({len(eval_dataset) if eval_dataset is not None else 0} cells)")
    print(f"Expression values enabled: {config.use_expression_values}")
    print("Cluster/Lineage auxiliary pooling: HVG tokens when auxiliary_gene_mask is present")
    print(
        "Auxiliary losses: "
        f"Cluster={config.cluster_auxiliary_loss_weight}, "
        f"Lineage={config.lineage_auxiliary_loss_weight}, "
        f"Study adversarial={config.study_adversarial_loss_weight}, "
        f"orig.ident adversarial={config.orig_ident_adversarial_loss_weight}, "
        f"multiome adversarial={config.multiome_adversarial_loss_weight}"
    )
    print(f"PCW losses: value={config.pcw_auxiliary_loss_weight}, rank={config.pcw_rank_loss_weight}, stats={pcw_stats or 'disabled'}")
    print(f"Checkpoint save_steps: {save_steps}; eval_steps: {eval_steps if eval_dataset is not None else 'disabled'}; warmup_steps: {warmup_steps}")
    print(f"Output: {training_output_dir}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=AtlasMambaCollator(token_dictionary=token_dictionary, mlm_probability=mlm_probability, pcw_column=pcw_column),
    )
    resume_from_checkpoint = os.environ.get("GENEFORMER_RESUME_FROM_CHECKPOINT", "").strip()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint or None)
    trainer.save_model(str(training_output_dir / "models"))

    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_log_history(
        trainer.state.log_history,
        metrics_dir / "trainer_log_history.csv",
        metrics_dir / "trainer_log_history.json",
    )
    latest_model_file = Path(os.environ.get("GENEFORMER_LATEST_MODEL_FILE", output_root / "latest_model_path.txt"))
    latest_model_file.parent.mkdir(parents=True, exist_ok=True)
    latest_model_file.write_text(str(training_output_dir / "models") + "\n")
    print(f"Wrote latest model marker: {latest_model_file}")
    if pcw_stats and hasattr(model, "pcw_scale"):
        pcw_stats["learned_pcw_scale"] = model.pcw_scale.parameters_as_dict()
        with (metrics_dir / "pcw_loss_scale_stats.json").open("w") as fp:
            json.dump(pcw_stats, fp, indent=2)


if __name__ == "__main__":
    main()
