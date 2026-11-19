"""Portable gene/regulon program resolution and summaries for release notebooks."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd


@lru_cache(maxsize=32)
def _gene_lookup(statistics, mapping, stats_stamp, mapping_stamp):
    stats=pd.read_csv(statistics,keep_default_na=False)
    lookup={}
    for row in stats.itertuples():
        if str(row.token)=='':continue
        token=int(row.token)
        for name in (str(row.gene),str(row.ensembl_id)):
            if name:lookup[name]=token
    if mapping is not None:
        aliases=pd.read_csv(mapping,sep='\t',keep_default_na=False)
        for row in aliases.itertuples(index=False,name=None):
            if len(row)>=2 and str(row[1]) in lookup:lookup.setdefault(str(row[0]),lookup[str(row[1])])
    return lookup


def gene_program(genes, statistics, mapping=None):
    """Resolve unique symbols/Ensembl IDs against the model's released vocabulary."""
    statistics=Path(statistics).resolve()
    mapping=Path(mapping).resolve() if mapping is not None else None
    stamp=lambda p:(p.stat().st_size,p.stat().st_mtime_ns) if p is not None else None
    lookup=_gene_lookup(statistics,mapping,stamp(statistics),stamp(mapping))
    requested=list(dict.fromkeys(str(x) for x in genes if pd.notna(x)))
    resolved={g:lookup[g] for g in requested if g in lookup}
    tokens=sorted(set(resolved.values()))
    if not tokens:raise ValueError('No requested genes map to this model')
    return dict(tokens=tokens,requested_genes=requested,resolved_genes=resolved,
                missing_genes=[g for g in requested if g not in resolved])


def regulon_programs(table, statistics, mapping=None, source='scenic', names=None, include_tf=True):
    """Resolve joint deletion sets; CollecTRI MOR is not a bulk fold change."""
    if source not in ('scenic','collectri'):raise ValueError('source must be scenic or collectri')
    frame=pd.read_csv(table,sep='\t')
    required={'tf','target'}|({'eRegulon_name'} if source=='scenic' else set())
    if not required.issubset(frame):raise ValueError('Missing columns: '+str(required-set(frame)))
    frame=frame.dropna(subset=list(required))
    key='eRegulon_name' if source=='scenic' else 'tf'
    programs={}
    selected=set(names) if names is not None else None
    for name,group in frame.groupby(key,sort=True):
        if selected is not None and name not in selected:continue
        tfs=group.tf.astype(str).unique()
        if len(tfs)!=1:raise ValueError('Regulon must contain one TF: '+str(name))
        genes=group.target.astype(str).tolist()
        if include_tf:genes=[tfs[0]]+genes
        programs[str(name)]=gene_program(genes,statistics,mapping)
        programs[str(name)].update(source=source,tf=tfs[0],include_tf=include_tf)
    if selected is not None and selected-set(programs):
        raise ValueError('Unknown or unresolved regulons: '+str(sorted(selected-set(programs))))
    if not programs:raise ValueError('No regulons selected')
    return programs


def no_op_controls(model, repo, rows, baseline_path, output, batch_size):
    """Exercise the real editor and predictor on empty/absent deletion controls."""
    from release_utils import delete_tokens,perturb
    absent=int(model.config.vocab_size)+1
    for tokens in ([],[absent]):
        edited=delete_tokens(rows,tokens)
        assert edited==rows,'No-op editing changed the input'
        result=perturb(model,repo,rows,baseline_path,edited,output,batch_size)
        np.testing.assert_allclose(result['prediction_delta'],0,rtol=0,atol=1e-6)
        np.testing.assert_allclose(result['embedding_l2'],0,rtol=0,atol=1e-6)


def summarize_age(result, metadata, present):
    """Mean signed effects per donor and equal-donor summaries; retain zero coverage."""
    required={'cell_id','donor_id','true_log2_PCW'}
    if not required.issubset(metadata):raise ValueError('Missing age metadata columns')
    data=metadata.reset_index(drop=True).copy()
    data['delta_log2_PCW']=np.asarray(result['prediction_delta']).reshape(-1)
    data['embedding_l2']=result['embedding_l2'];data['target_present']=present
    donors=data.groupby('donor_id',sort=True).agg(
        true_mean_log2_PCW=('true_log2_PCW','mean'),mean_delta_log2_PCW=('delta_log2_PCW','mean'),
        mean_embedding_l2=('embedding_l2','mean'),fraction_target_present=('target_present','mean'),n_cells=('cell_id','size')).reset_index()
    summary=dict(mean_delta_log2_PCW=float(donors.mean_delta_log2_PCW.mean()),
                 mean_absolute_donor_delta=float(donors.mean_delta_log2_PCW.abs().mean()),
                 n_donors=len(donors),fraction_target_present=float(np.mean(present)))
    return data,donors,summary


def summarize_classifier(result, metadata, present, labels):
    """Class-probability deltas and pooled embedding L2, grouped by annotation."""
    delta=np.asarray(result['prediction_delta'])
    if delta.ndim!=2 or delta.shape[1]!=len(labels):raise ValueError('Class map does not match predictions')
    data=metadata.reset_index(drop=True).copy()
    data['embedding_l2']=result['embedding_l2'];data['target_present']=present
    for i,label in enumerate(labels):data['delta_prob:'+str(label)]=delta[:,i]
    if 'cell_type' in data:
        ids={str(label):i for i,label in enumerate(labels)}
        unknown=set(data.cell_type.astype(str))-set(ids)
        if unknown:raise ValueError('Cell annotations differ from saved broad labels: '+str(unknown))
        data['annotated_class_probability_delta']=[delta[i,ids[str(x)]] for i,x in enumerate(data.cell_type)]
    columns=[c for c in data if c.startswith('delta_prob:')]+['embedding_l2','target_present']
    if 'annotated_class_probability_delta' in data:columns+=['annotated_class_probability_delta']
    grouping=[c for c in ('donor_id','cell_type') if c in data]
    groups=data.groupby(grouping)[columns].mean().reset_index() if grouping else data[columns].mean().to_frame().T
    return data,groups
