"""CPU statistics for the all-gene deletion screen; no age-program assignment."""
import numpy as np
import pandas as pd


def summarize_gene(gene, meta, baseline, present_indices, perturbed, l2, cosine):
    """Absent tokens have exactly zero effect; absent genes remain not estimable."""
    n = len(meta)
    present_indices = np.asarray(present_indices, dtype=np.int64)
    if len(np.unique(present_indices)) != len(present_indices):
        raise ValueError('Duplicate present-cell index')
    if len(present_indices) and (present_indices.min() < 0 or present_indices.max() >= n):
        raise ValueError('Present-cell index out of range')
    arrays = [np.asarray(x, dtype=float) for x in (baseline, perturbed, l2, cosine)]
    baseline, perturbed, l2, cosine = arrays
    if len(baseline) != n or any(len(a) != len(present_indices) for a in arrays[1:]):
        raise ValueError('Predictions/embedding statistics must align with cell indices')
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Nonfinite predictions or embedding statistics')
    present = np.zeros(n, dtype=bool)
    present[present_indices] = True
    delta = np.zeros(n, dtype=float)
    delta[present_indices] = perturbed-baseline[present_indices]
    shifts = np.zeros(n, dtype=float)
    shifts[present_indices] = l2
    similarities = np.ones(n, dtype=float)
    similarities[present_indices] = cosine
    evaluated = bool(len(present_indices))
    status = 'evaluated' if evaluated else ('not_in_token_dictionary' if gene.get('token') in (None, '') else 'not_in_test_input')
    common = dict(gene)
    common.update(perturbation_label=gene['gene'], input_gene=gene['gene'], perturbation_kind='single_gene',
                  perturb_type='delete', status=status)
    def stats(idx):
        p = present[idx]
        d = delta[idx]
        return dict(n_cells=int(len(idx)), n_cells_with_perturbed_gene=int(p.sum()),
            fraction_cells_with_perturbed_gene=float(p.mean()),
            base_pred_mean=float(baseline[idx].mean()),
            perturbed_pred_mean=float((baseline[idx]+d).mean()) if evaluated else np.nan,
            delta_pred_mean=float(d.mean()) if evaluated else np.nan,
            delta_pred_median=float(np.median(d)) if evaluated else np.nan,
            delta_pred_std=float(d.std()) if evaluated else np.nan,
            present_delta_pred_mean=float(d[p].mean()) if p.any() else np.nan,
            present_delta_pred_median=float(np.median(d[p])) if p.any() else np.nan,
            embedding_l2_shift_mean=float(shifts[idx].mean()) if evaluated else np.nan,
            present_embedding_l2_shift_mean=float(shifts[idx][p].mean()) if p.any() else np.nan,
            embedding_cosine_original_perturbed_mean=float(similarities[idx].mean()) if evaluated else np.nan,
            present_embedding_cosine_original_perturbed_mean=float(similarities[idx][p].mean()) if p.any() else np.nan)
    donor_rows = []
    for donor, indexes in meta.groupby('sample_id', sort=True).indices.items():
        idx = np.asarray(indexes, dtype=int)
        part = meta.iloc[idx]
        row = dict(common, sample_id=donor, **stats(idx))
        row['true_mean'] = float(part.true_value.mean())
        for column in ['AgeGroup', 'PCW', 'log2_PCW', 'Location', 'donor_id', 'SampleID']:
            if column in part:
                unique = part[column].dropna().astype(str).unique()
                if column in ['AgeGroup', 'PCW', 'log2_PCW', 'donor_id'] and len(unique) > 1:
                    raise ValueError(f'Conflicting {column} within donor {donor}')
                row[column] = ','.join(sorted(unique))
        donor_rows.append(row)
    donor_frame = pd.DataFrame(donor_rows)
    summary = dict(common, **stats(np.arange(n)))
    summary.update(n_donors=len(donor_frame),
        n_donors_with_perturbed_gene=int((donor_frame.n_cells_with_perturbed_gene>0).sum()),
        equal_donor_delta_pred_mean=float(donor_frame.delta_pred_mean.mean()) if evaluated else np.nan,
        median_donor_delta_pred_mean=float(donor_frame.delta_pred_mean.median()) if evaluated else np.nan,
        n_donors_older=int((donor_frame.delta_pred_mean>1e-7).sum()) if evaluated else np.nan,
        n_donors_younger=int((donor_frame.delta_pred_mean< -1e-7).sum()) if evaluated else np.nan)
    sparse = pd.DataFrame(dict(gene_index=int(gene['gene_index']), cell_index=present_indices,
                               delta_pred_value=delta[present_indices], embedding_l2_shift=l2,
                               embedding_cosine_original_perturbed=cosine))
    return summary, donor_frame, sparse


def build_token_postings(rows, max_length, pad_token_id):
    """Index the actual retained inputs, once per unique token per cell."""
    token_chunks, cell_chunks = [], []
    for index, row in enumerate(rows):
        tokens = np.asarray(sorted({int(t) for t in row['input_ids'][:max_length] if int(t) != pad_token_id}), dtype=np.int32)
        token_chunks.append(tokens)
        cell_chunks.append(np.full(len(tokens), index, dtype=np.int32))
    if not token_chunks:
        raise ValueError('No cells to index')
    tokens = np.concatenate(token_chunks)
    cells = np.concatenate(cell_chunks)
    order = np.argsort(tokens, kind='stable')
    tokens, cells = tokens[order], cells[order]
    unique, counts = np.unique(tokens, return_counts=True)
    return dict(tokens=unique, offsets=np.r_[0,np.cumsum(counts)], cells=cells)


def token_cells(postings, token):
    index = np.searchsorted(postings['tokens'], token)
    if index == len(postings['tokens']) or postings['tokens'][index] != token:
        return np.empty(0, dtype=np.int64)
    return postings['cells'][postings['offsets'][index]:postings['offsets'][index+1]]
