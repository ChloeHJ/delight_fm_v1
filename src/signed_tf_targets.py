"""Signed TF-response edits in the v11 tokenization scale (CPU only)."""
import csv
import json
import math
import pickle
import struct
from pathlib import Path


def resolve_targets(target_csv, var_csv, mapping_tsv, token_dictionary, tf="BHLHE40", value_clip=5.,
                    perturb_type="overexpress", tf_overexpress_value=5., max_length=2048):
    if perturb_type not in {"overexpress", "delete"}:
        raise ValueError("TF mode must be overexpress or delete")
    if not math.isfinite(value_clip) or value_clip <= 0 or max_length < 1:
        raise ValueError("Expression clipping and context length must be positive")
    if not math.isfinite(tf_overexpress_value) or not 0 < tf_overexpress_value <= value_clip:
        raise ValueError("TF overexpression value must be positive and within the clipping range")
    direction = 1 if perturb_type == "overexpress" else -1
    with Path(target_csv).open() as handle:
        reader = csv.DictReader(handle)
        if not {"Gene", "bulk_log2FC", "bulk_padj"}.issubset(reader.fieldnames or []):
            raise ValueError("Target CSV needs Gene, bulk_log2FC and bulk_padj")
        targets = list(reader)
    if not targets:
        raise ValueError("Target table is empty")
    with Path(var_csv).open() as handle:
        variables = {row["gene"]: row for row in csv.DictReader(handle)}
    with Path(mapping_tsv).open() as handle:
        mapping = {row["gene_symbol"]: row["ensembl_id"].split(".")[0]
                   for row in csv.DictReader(handle, delimiter="\t")}
    with Path(token_dictionary).open("rb") as handle:
        dictionary = pickle.load(handle)

    def token_for(symbol):
        value = dictionary.get(symbol, dictionary.get(mapping.get(symbol)))
        return int(value) if value is not None else None

    tf_token = token_for(tf)
    if tf_token is None:
        raise ValueError(f"TF {tf} has no dictionary token")
    coverage, plan, seen, token_genes = [], [], set(), {}
    for row in targets:
        gene = row["Gene"].strip()
        if not gene or gene in seen:
            raise ValueError(f"Empty or duplicated target symbol: {gene}")
        seen.add(gene)
        fc, padj = float(row["bulk_log2FC"]), float(row["bulk_padj"])
        if not math.isfinite(fc) or not math.isfinite(padj) or not 0 <= padj <= 1:
            raise ValueError(f"Invalid logFC or adjusted p-value for {gene}")
        token = token_for(gene)
        status = "mapped"
        if gene == tf:
            status = "tf_edited_separately"
        elif fc == 0:
            status = "zero_logfc"
        elif gene not in variables:
            status = "not_in_model_features"
        elif token is None:
            status = "no_dictionary_token"
        coverage.append(dict(gene=gene, bulk_log2FC=fc, bulk_padj=padj, token=token, status=status,
                             response_direction="up" if direction*fc > 0 else "down" if direction*fc < 0 else "unchanged"))
        if status != "mapped":
            continue
        if token in token_genes:
            raise ValueError(f"Targets {gene} and {token_genes[token]} share token {token}")
        token_genes[token] = gene
        mean = float(variables[gene]["mean_log_normalized"])
        sd = max(float(variables[gene]["sd_log_normalized"]), 1e-4)
        if not math.isfinite(mean) or not math.isfinite(sd) or mean < 0 or sd <= 0:
            raise ValueError(f"Invalid tokenization statistics for {gene}")
        plan.append(dict(gene=gene, token=token, bulk_log2FC=fc, mean=mean, sd=sd))
    if not plan:
        raise ValueError("No signed targets map to model features")
    tf_var = variables.get(tf, {})
    tf_aux = any(str(tf_var.get(column, "false")).lower() == "true"
                 for column in ["in_global_hvg", "in_any_lineage_hvg"])
    return dict(version=2, signed_edit_version="float32_v2", tf=tf, tf_token=tf_token, value_clip=float(value_clip), targets=plan,
                perturb_type=perturb_type, response_sign=direction, tf_edit=True,
                tf_overexpress_value=float(tf_overexpress_value), tf_auxiliary_mask=int(tf_aux), max_length=max_length,
                direction=f"TF {perturb_type}: target log2 fold change = {direction:+d} * strength * bulk_log2FC",
                expression_edit="log1p(expm1(max(0, mean + sd*z)) * 2**(response_sign*strength*bulk_log2FC))",
                limitations="Only target tokens present in each input are edited; absent/truncated targets are not inserted. "
                             "Expression reconstruction is approximate where original z-scores were clipped."), coverage


def make_programs(plan, strengths):
    strengths = [float(value) for value in strengths]
    if not strengths or len(set(strengths)) != len(strengths) or any(not math.isfinite(x) or x <= 0 for x in strengths):
        raise ValueError("Signed target strengths must be unique, finite positive values")
    tf, tf_token = plan["tf"], int(plan["tf_token"])
    tokens = sorted({int(row["token"]) for row in plan["targets"]} | {tf_token})
    common = dict(input_gene=tf, token=tf_token, n_targets=len(plan["targets"]),
                  age_contribution_program="unspecified", target_source="bound_bulk_DEGs")
    mode = plan["perturb_type"]
    programs = [dict(common, perturbation_label=f"{tf}_{mode}_signed_strength_{strength:g}",
                     perturbation_kind="signed_target_regulon", edit_mode=f"signed_expression_and_tf_{mode}",
                     tokens=tokens, n_tokens=len(tokens), signed_target_plan=plan, signed_strength=strength,
                     target_genes=",".join(row["gene"] for row in plan["targets"])) for strength in strengths]
    programs.extend([
        dict(common, perturbation_label=f"{tf}_TF_only_{mode}", perturbation_kind="single_gene",
             edit_mode=f"tf_only_{mode}", tokens=[tf_token], n_tokens=1, target_genes=tf),
        dict(common, perturbation_label=f"{tf}_targets_only_signed_{mode}", perturbation_kind="signed_target_regulon",
             edit_mode="signed_targets_without_tf_edit", tokens=[token for token in tokens if token != tf_token],
             n_tokens=len(tokens)-1, signed_target_plan=dict(plan, tf_edit=False), signed_strength=1.,
             target_genes=",".join(row["gene"] for row in plan["targets"]))])
    return programs


def _softplus(value):
    return value + math.log1p(math.exp(-value)) if value > 0 else math.log1p(math.exp(value))


def _float32(value):
    """Round a sort key to the precision used by the inference collator."""
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def edit_signed_rows(rows, plan, strength):
    """Edit existing tokens; preserve aligned masks and rerank by the new z-score."""
    if plan.get('target_edit') is not None:
        raise ValueError('Only proportional signed bulk-response edits are supported in v12')
    version = plan.get("signed_edit_version")
    if version != "float32_v2":
        raise ValueError(f"Unknown signed edit version: {version}")
    order = plan.get("target_order", "rerank")
    if order not in {"rerank", "preserve"}:
        raise ValueError(f"Unknown target ordering: {order}")
    parameters = {int(row["token"]): row for row in plan["targets"]}
    tf_token, clip = int(plan["tf_token"]), float(plan["value_clip"])
    edited, audits = [], []
    for row in rows:
        ids, values = list(row["input_ids"]), list(row["expression_values"])
        masks = list(row.get("auxiliary_gene_mask", [0] * len(ids)))
        if len(ids) != len(values) or len(ids) != len(masks):
            raise ValueError("Tokens, expression values and auxiliary masks must be aligned")
        audit = dict(n_target_tokens_present=0, n_target_tokens_upregulated=0,
                     n_target_tokens_downregulated=0, n_target_tokens_source_clipped=0,
                     n_target_tokens_result_clipped=0, tf_token_present=tf_token in ids,
                     tf_token_inserted=False, n_tokens_truncated=0)
        remaining = []
        tf_edit = plan.get("tf_edit", True)
        sort_required = tf_edit and plan["perturb_type"] == "overexpress"
        if tf_edit and plan["perturb_type"] == "overexpress":
            tf_mask = masks[ids.index(tf_token)] if tf_token in ids else plan["tf_auxiliary_mask"]
            remaining.append((tf_token, plan["tf_overexpress_value"], tf_mask))
            audit["tf_token_inserted"] = tf_token not in ids
        for token, value, mask in zip(ids, values, masks):
            if int(token) == tf_token and tf_edit:
                continue
            if int(token) in parameters and strength != 0:
                target = parameters[int(token)]
                audit["n_target_tokens_present"] += 1
                audit["n_target_tokens_source_clipped"] += int(abs(value) >= clip - 1e-6)
                if float(target["bulk_log2FC"]) == 0.:
                    z = value  # Exact identity; avoid reconstruction roundoff.
                else:
                    x = max(0., target["mean"] + target["sd"] * value)
                    if x > 0:
                        log_abundance = math.log(math.expm1(x)) if x < 50 else x + math.log1p(-math.exp(-x))
                        changed = _softplus(log_abundance + plan["response_sign"] * strength * target["bulk_log2FC"] * math.log(2))
                    else:
                        changed = 0.
                    z = (changed - target["mean"]) / target["sd"]
                new_value = max(-clip, min(clip, z))
                model_changed = _float32(new_value) != _float32(value)
                sort_required = sort_required or model_changed
                if not model_changed:
                    new_value = value  # Preserve exact input for model-level no-ops.
                audit["n_target_tokens_result_clipped"] += int(abs(z) >= clip)
                audit["n_target_tokens_upregulated"] += int(new_value > value + 1e-7)
                audit["n_target_tokens_downregulated"] += int(new_value < value - 1e-7)
                value = new_value
            remaining.append((token, value, mask))
        # Preserve ties at model precision, starting from the original token order.
        if order == "rerank":
            if sort_required:
                remaining.sort(key=lambda item: _float32(item[1]), reverse=True)
        audit["n_tokens_truncated"] = max(0, len(remaining)-plan["max_length"])
        remaining = remaining[:plan["max_length"]]
        new_row = dict(row, input_ids=[item[0] for item in remaining],
                       expression_values=[item[1] for item in remaining], length=len(remaining))
        if "auxiliary_gene_mask" in row:
            new_row["auxiliary_gene_mask"] = [item[2] for item in remaining]
        edited.append(new_row)
        audits.append(audit)
    return edited, audits


def load_programs(path, strengths):
    return make_programs(json.loads(Path(path).read_text()), [float(x) for x in strengths.split(",") if x.strip()])
