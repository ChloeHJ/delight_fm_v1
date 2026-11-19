"""Bind exact final donor splits, or rebuild inputs from RNA RDS and fixed donors."""
from pathlib import Path
import json
import os
from paths import RUN,SRC
from common import config,inputs,atomic_json,run


def safe_link(source,target):
    source=Path(source).resolve();target=Path(target)
    if not source.exists():raise FileNotFoundError(source)
    if source==target.resolve():return
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink():
        if target.resolve()!=source:raise ValueError('Different existing input link: '+str(target))
    elif not target.exists():target.symlink_to(source,target_is_directory=source.is_dir())
    else:raise ValueError('Existing real input directory; validate it instead of linking: '+str(target))


def prepare(group):
    record=inputs()['models'][group];root=Path(record['data_root'])
    for folder in ['exported','tokenized']:
        safe_link(root/folder/group,RUN/folder/group)
    return validate(group)


def validate(group):
    import numpy as np
    import pandas as pd
    from datasets import load_from_disk
    wanted=pd.read_csv(RUN/'donors'/f'{group}.csv',dtype={'sample_id':str})
    if wanted.sample_id.duplicated().any():raise ValueError('Duplicate fixed donor assignment')
    assignment=wanted.set_index('sample_id').SplitDonor
    frames=[];counts={}
    columns=['cell_id','sample_id','AgeGroup','PCW','log2_PCW','length']
    for split in ['train','test']:
        ds=load_from_disk(str(RUN/'tokenized'/group/(split+'.dataset')))
        frame=ds.select_columns(columns).to_pandas();frame['sample_id']=frame.sample_id.astype(str)
        for column in ['PCW','log2_PCW','length']:frame[column]=pd.to_numeric(frame[column],errors='raise')
        if frame.empty or frame.cell_id.duplicated().any():raise ValueError('Empty or duplicate cells')
        if not frame.sample_id.map(assignment).eq(split.upper()).all():raise ValueError('Donor assignment mismatch: '+group)
        if not np.allclose(frame.log2_PCW,np.log2(pd.to_numeric(frame.PCW))):raise ValueError('Age target mismatch')
        if not frame.length.between(1,config()['sweep']['max_length']).all():raise ValueError('Invalid token sequence length')
        frame['SplitDonor']=split.upper();frames.append(frame)
        counts[split]=dict(cells=len(frame),donors=frame.sample_id.nunique())
    allcells=pd.concat(frames,ignore_index=True)
    if allcells.cell_id.duplicated().any() or allcells.groupby('sample_id').SplitDonor.nunique().max()!=1:
        raise ValueError('Train/test leakage')
    expected=wanted.set_index('sample_id').n_cells.sort_index()
    pd.testing.assert_series_equal(allcells.groupby('sample_id').size().sort_index(),expected,check_names=False)
    record=inputs()['models'][group]
    if allcells.AgeGroup.isin(record['excluded_age_groups']).any():raise ValueError('Excluded ages present')
    train=frames[0].groupby('AgeGroup').size().reindex(allcells.AgeGroup.unique(),fill_value=0)
    if train.lt(record['minimum_training_cells_per_age']).any():raise ValueError('Minimum training age coverage failed')
    age=allcells.groupby(['AgeGroup','SplitDonor']).agg(n_cells=('cell_id','size'),n_donors=('sample_id','nunique'),mean_PCW=('PCW','mean')).reset_index()
    (RUN/'diagnostics').mkdir(exist_ok=True)
    age.sort_values('mean_PCW').to_csv(RUN/'diagnostics'/f'{group}_age_counts.csv',index=False)
    atomic_json(RUN/'diagnostics'/f'{group}_inputs.json',dict(group=group,counts=counts,donor_overlap=0,
        minimum_training_age_cells=int(train.min()),source=record))
    print(json.dumps(dict(group=group,**counts),indent=2),flush=True)
    return counts


def rebuild(group):
    record=inputs()['models'][group]
    for folder in ['exported','tokenized']:
        p=RUN/folder/group
        if p.exists() or p.is_symlink():raise ValueError('Rebuild requires an empty input destination: '+str(p))
    env=dict(os.environ,ATLAS_MAMBA_INPUT_RDS=record['rds'],EXPORT_DIR=str(RUN/'exported'/group),
        ATLAS_MAMBA_LINEAGE_GROUP=group,V12_DONORS_CSV=str(RUN/'donors'/f'{group}.csv'),
        V12_EXCLUDED_AGES=json.dumps(record['excluded_age_groups']))
    run(['Rscript',SRC/'prepare_inputs.R'],env)
    run([config()['model_python'],SRC/'build_atlas_hf_dataset.py','--export-dir',RUN/'exported'/group,
        '--token-dictionary',RUN/'data/reference/token_dictionary.pkl','--gene-mapping-tsv',RUN/'data/reference/gene_symbol_to_ensembl.tsv',
        '--output-dir',RUN/'tokenized'/group,'--max-length',str(config()['sweep']['max_length']),'--seed','42'])
    validate(group)
