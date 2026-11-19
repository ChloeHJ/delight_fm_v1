#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from datasets import Dataset, DatasetDict


def clean_value(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value)
    if text.lower() in {"nan", "none", "na", "n/a", "null", ""}:
        return None
    return text


def sanitize_h5ad_frame(frame: pd.DataFrame) -> pd.DataFrame:
    sanitized = frame.copy()
    for column in sanitized.columns:
        series = sanitized[column]
        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            sanitized[column] = series.map(clean_value).fillna("").astype(str)
    return sanitized


def clean_symbol(symbol: str) -> str:
    symbol = str(symbol).strip()
    if "." in symbol and not symbol.startswith("ENSG"):
        return symbol.split(".", 1)[0]
    return symbol


def parse_gtf_attrs(attr_text: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for match in re.finditer(r'(\S+) "([^"]+)"', attr_text)}


def load_gencode_mapping(gtf_gz: Path) -> dict[str, str]:
    mapping = {}
    with gzip.open(gtf_gz, "rt") as fp:
        for line in fp:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            attrs = parse_gtf_attrs(fields[8])
            gene_name = attrs.get("gene_name")
            gene_id = attrs.get("gene_id")
            if not gene_name or not gene_id:
                continue
            mapping.setdefault(gene_name, re.sub(r"\.\d+$", "", gene_id))
    return mapping


def load_symbol_mapping(mapping_tsv: Path) -> dict[str, str]:
    mapping = pd.read_csv(mapping_tsv, sep="\t")
    required = {"gene_symbol", "ensembl_id"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Mapping file missing columns: {sorted(missing)}")
    mapping = mapping[["gene_symbol", "ensembl_id"]].dropna()
    mapping["gene_symbol"] = mapping["gene_symbol"].astype(str)
    mapping["ensembl_id"] = mapping["ensembl_id"].astype(str).str.replace(r"\.\d+$", "", regex=True)
    mapping = mapping.drop_duplicates("gene_symbol", keep="first")
    return dict(zip(mapping["gene_symbol"], mapping["ensembl_id"]))


def fill_ensembl_ids(var: pd.DataFrame, mapping: dict[str, str]) -> pd.Series:
    existing = pd.Series([None] * len(var), index=var.index, dtype="object")
    if "ensembl_gene_id" in var.columns:
        existing = var["ensembl_gene_id"].map(clean_value)
        existing = existing.where(existing.isna(), existing.astype(str).str.replace(r"\.\d+$", "", regex=True))

    mapped = var["gene"].astype(str).map(lambda gene: mapping.get(clean_symbol(gene)))
    return existing.fillna(mapped)


def write_h5ad(matrix, obs: pd.DataFrame, var: pd.DataFrame, output_h5ad: Path) -> None:
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata_var = sanitize_h5ad_frame(var)
    adata_var.index = adata_var["gene"].astype(str)
    adata_obs = sanitize_h5ad_frame(obs)
    adata_obs.index = adata_obs["cell_id"].astype(str)
    adata = ad.AnnData(X=matrix.T.tocsr(), obs=adata_obs, var=adata_var)
    adata.write_h5ad(output_h5ad)


def donor_split(obs: pd.DataFrame, eval_fraction: float, seed: int) -> np.ndarray:
    if "SplitDonor" in obs.columns:
        split = obs["SplitDonor"].astype(str).str.upper()
        if split.isin(["TRAIN", "TEST"]).any():
            return np.where(split == "TEST", "test", "train")

    donor_col = next((c for c in ("sample_id", "donor_id", "SampleID") if c in obs.columns), None)
    if donor_col is None:
        raise ValueError("A donor column is required for donor-disjoint splitting")
    donors = obs[donor_col].map(clean_value)
    if donors.isna().any():
        raise ValueError(f"Missing donor identifiers in {donor_col}")
    donors = donors.astype(str)
    unique_donors = np.array(sorted(donors.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_donors)
    n_eval = max(1, int(round(len(unique_donors) * eval_fraction)))
    eval_donors = set(unique_donors[:n_eval])
    return np.where(donors.isin(eval_donors), "test", "train")


def encode_category(values: pd.Series) -> tuple[list[int], dict[str, int]]:
    labels = sorted({clean_value(v) for v in values if clean_value(v) is not None})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    return [label_to_id.get(clean_value(v), -1) for v in values], label_to_id


def make_row(
    col,
    cell_idx: int,
    obs_row: pd.Series,
    genes: list[str],
    token_keys: list[str],
    token_dictionary: dict,
    gene_mean: np.ndarray,
    gene_sd: np.ndarray,
    auxiliary_gene_mask: np.ndarray,
    max_length: int,
    value_clip: float,
    metadata_columns: list[str],
):
    start, end = col.indptr[cell_idx], col.indptr[cell_idx + 1]
    gene_idx = col.indices[start:end]
    values = col.data[start:end].astype(np.float32)
    keep = values > 0
    gene_idx = gene_idx[keep]
    values = values[keep]

    tokens = []
    token_values = []
    token_auxiliary_mask = []
    for idx, value in zip(gene_idx, values):
        token = token_dictionary.get(token_keys[int(idx)])
        if token is None:
            token = token_dictionary.get(genes[int(idx)])
        if token is None:
            continue
        scaled = (float(value) - float(gene_mean[int(idx)])) / float(gene_sd[int(idx)])
        tokens.append(int(token))
        token_values.append(float(np.clip(scaled, -value_clip, value_clip)))
        token_auxiliary_mask.append(int(auxiliary_gene_mask[int(idx)]))

    if tokens:
        order = np.argsort(np.asarray(token_values))[::-1][:max_length]
        input_ids = [tokens[i] for i in order]
        expression_values = [token_values[i] for i in order]
        auxiliary_gene_mask_values = [token_auxiliary_mask[i] for i in order]
    else:
        input_ids = []
        expression_values = []
        auxiliary_gene_mask_values = []

    row = {
        "cell_id": str(obs_row["cell_id"]),
        "input_ids": input_ids,
        "expression_values": expression_values,
        "auxiliary_gene_mask": auxiliary_gene_mask_values,
        "length": len(input_ids),
    }
    for column in metadata_columns:
        if column in obs_row.index:
            row[column] = clean_value(obs_row[column])
    return row


def main():
    parser = argparse.ArgumentParser(description="Build value-aware Mamba HF datasets from Seurat MatrixMarket export.")
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--token-dictionary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--value-clip", type=float, default=5.0)
    parser.add_argument("--gene-mapping-tsv", default=None)
    parser.add_argument(
        "--gencode-gtf-gz",
        default=None,
    )
    parser.add_argument("--h5ad-output", default=None)
    parser.add_argument("--min-tokenized-cells", type=int, default=1)
    parser.add_argument(
        "--metadata-columns",
        default="Lineage,OriginalLineage,Cluster,BroadCellType,CellType,study,orig.ident,multiome,Multiome,donor_id,SampleID,sample_id,batch,Location,location,AgeGroup,age,PCW,log2_PCW,lineage_cluster_id,lineage_seurat_cluster",
    )
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    output_dir = Path(args.output_dir)
    metadata_columns = [x.strip() for x in args.metadata_columns.split(",") if x.strip()]

    with open(args.token_dictionary, "rb") as fp:
        token_dictionary = pickle.load(fp)

    obs = pd.read_csv(export_dir / "obs.csv.gz")
    var = pd.read_csv(export_dir / "var.csv")
    genes = var["gene"].astype(str).tolist()
    mapping = {}
    if args.gene_mapping_tsv:
        mapping = load_symbol_mapping(Path(args.gene_mapping_tsv))
        print(f"Loaded symbol mapping TSV: {args.gene_mapping_tsv} ({len(mapping)} symbols)")
    elif args.gencode_gtf_gz and Path(args.gencode_gtf_gz).exists():
        mapping = load_gencode_mapping(Path(args.gencode_gtf_gz))
        print(f"Loaded GENCODE mapping: {args.gencode_gtf_gz} ({len(mapping)} symbols)")
    elif "ensembl_gene_id" not in var.columns or var["ensembl_gene_id"].map(clean_value).isna().all():
        print("WARNING: no gene mapping file found and var.csv has no usable ensembl_gene_id values.")

    if mapping:
        var["ensembl_gene_id"] = fill_ensembl_ids(var, mapping)

    if "ensembl_gene_id" in var.columns:
        token_keys = [
            str(value) if clean_value(value) is not None else genes[idx]
            for idx, value in enumerate(var["ensembl_gene_id"].tolist())
        ]
    else:
        token_keys = genes
    gene_mean = var["mean_log_normalized"].to_numpy(dtype=np.float32)
    gene_sd = np.maximum(var["sd_log_normalized"].to_numpy(dtype=np.float32), 1e-4)
    auxiliary_gene_mask = (
        var.get("in_global_hvg", pd.Series([False] * len(var))).fillna(False).astype(bool)
        | var.get("in_any_lineage_hvg", pd.Series([False] * len(var))).fillna(False).astype(bool)
    ).to_numpy()

    matrix = scipy.io.mmread(export_dir / "log_normalized_gene_by_cell.mtx").tocsc()
    if matrix.shape != (len(genes), len(obs)):
        raise ValueError(f"Matrix shape {matrix.shape} does not match genes/cells: {len(genes)} x {len(obs)}")

    obs["split"] = donor_split(obs, args.eval_fraction, args.seed)
    if obs.groupby("sample_id")["split"].nunique().max() > 1:
        raise ValueError("SplitDonor assigns the same donor to both train and test")
    if "PCW" in obs.columns:
        obs["log2_PCW"] = np.log2(pd.to_numeric(obs["PCW"], errors="coerce").where(lambda x: x > 0))
    if "sample_id" not in obs.columns:
        if "donor_id" in obs.columns:
            obs["sample_id"] = obs["donor_id"]
        elif "SampleID" in obs.columns:
            obs["sample_id"] = obs["SampleID"]
    if "location" not in obs.columns and "Location" in obs.columns:
        obs["location"] = obs["Location"]
    if "multiome" not in obs.columns and "Multiome" in obs.columns:
        obs["multiome"] = obs["Multiome"]
    obs["Cluster_id"], cluster_label_to_id = encode_category(obs["Cluster"])
    obs["Lineage_id"], lineage_label_to_id = encode_category(obs["Lineage"])
    obs["study_id"], study_label_to_id = encode_category(obs["study"] if "study" in obs.columns else pd.Series(["missing"] * len(obs)))
    obs["orig_ident_id"], orig_ident_label_to_id = encode_category(obs["orig.ident"] if "orig.ident" in obs.columns else pd.Series([None] * len(obs)))
    obs["multiome_id"], multiome_label_to_id = encode_category(obs["multiome"] if "multiome" in obs.columns else pd.Series([None] * len(obs)))

    metadata_columns = list(dict.fromkeys(metadata_columns + ["split", "Cluster_id", "Lineage_id", "study_id", "orig_ident_id", "multiome_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    h5ad_output = Path(args.h5ad_output) if args.h5ad_output else output_dir / "atlas_geneformer_ready.h5ad"
    write_h5ad(matrix, obs, var, h5ad_output)

    n_genes_with_dictionary_token = int(sum((key in token_dictionary) or (gene in token_dictionary) for key, gene in zip(token_keys, genes)))
    if n_genes_with_dictionary_token == 0:
        raise ValueError(
            "No exported genes match the token dictionary. "
            "Provide --gene-mapping-tsv or --gencode-gtf-gz so gene symbols can be mapped to Ensembl IDs."
        )

    from datasets.arrow_writer import ArrowWriter
    arrow_path = output_dir / "tokenized.arrow"
    writer = ArrowWriter(path=str(arrow_path), writer_batch_size=128)
    n_rows = 0
    for i in range(matrix.shape[1]):
        row = make_row(matrix, i, obs.iloc[i], genes, token_keys, token_dictionary,
                       gene_mean, gene_sd, auxiliary_gene_mask, args.max_length,
                       args.value_clip, metadata_columns)
        if row["length"]:
            writer.write(row)
            n_rows += 1
        if (i + 1) % 10000 == 0:
            print(f"Tokenized {i + 1}/{matrix.shape[1]} cells", flush=True)
    writer.finalize()
    if n_rows < args.min_tokenized_cells:
        raise ValueError(f"Only {n_rows} cells tokenized")
    dataset = Dataset.from_file(str(arrow_path))
    dataset_dict = DatasetDict({
        "train": dataset.filter(lambda row: row["split"] == "train"),
        "test": dataset.filter(lambda row: row["split"] == "test"),
    })

    dataset_dict["train"].save_to_disk(str(output_dir / "train.dataset"))
    dataset_dict["test"].save_to_disk(str(output_dir / "test.dataset"))
    if not all(len(ds) for ds in dataset_dict.values()):
        raise ValueError("Both train and test must contain tokenized cells")
    # The split datasets now own their Arrow files; discard the redundant
    # temporary combined Arrow file generated by this invocation.
    arrow_path.unlink()

    with (output_dir / "label_maps.json").open("w") as fp:
        json.dump(
            {
                "Cluster": cluster_label_to_id,
                "Lineage": lineage_label_to_id,
                "study": study_label_to_id,
                "orig.ident": orig_ident_label_to_id,
                "multiome": multiome_label_to_id,
                "n_genes": len(genes),
                "n_auxiliary_hvg_genes": int(auxiliary_gene_mask.sum()),
                "n_genes_with_dictionary_token": n_genes_with_dictionary_token,
                "n_cells_input": int(len(obs)),
                "n_cells_tokenized": int(len(dataset)),
                "split_counts": {key: int(value) for key, value in obs["split"].value_counts().to_dict().items()},
                "h5ad_output": h5ad_output.name,
                "h5ad_path_base": "label_maps_directory",
                "max_length": args.max_length,
                "value_definition": "expression_values are per-gene z-scored log-normalized expression, clipped symmetrically",
            },
            fp,
            indent=2,
        )

    print(f"Wrote train dataset: {output_dir / 'train.dataset'} ({len(dataset_dict['train'])} cells)")
    print(f"Wrote test dataset: {output_dir / 'test.dataset'} ({len(dataset_dict['test'])} cells)")
    print(f"Wrote h5ad: {h5ad_output}")
    print(f"Wrote label maps: {output_dir / 'label_maps.json'}")


if __name__ == "__main__":
    main()
