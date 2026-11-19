#!/usr/bin/env python3
"""Prepare, submit and merge an all-gene deletion age screen without plotting."""
import argparse
import csv
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import pickle
import shlex
import subprocess

from paths import RUN, SRC, TASK

GROUPS = ['epithelium_ileum', 'epithelium_colon', 'immune', 'stromal']
DEFAULT_OUTPUT = RUN/'outputs/perturb_age_all_genes'
SCHEMA_VERSION = 1


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    from common import atomic_json
    atomic_json(path,value)


def write_csv(path, records):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    temporary.replace(path)


def fingerprint(path):
    path=Path(path)
    if path.is_dir() and (path/'state.json').exists():
        state=json.loads((path/'state.json').read_text())
        files=[path/'state.json',path/'dataset_info.json']+[path/row['filename'] for row in state['_data_files']]
    else:
        files=sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    if not files or any(not p.exists() for p in files):
        raise FileNotFoundError(path)
    return [dict(path=str(p.resolve()),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in files]


def catalog(var_path, mapping_path, dictionary_path, max_genes=0):
    with mapping_path.open() as handle:
        mapping={row['gene_symbol']:row['ensembl_id'] for row in csv.DictReader(handle,delimiter='\t')}
    with dictionary_path.open('rb') as handle:
        tokens=pickle.load(handle)
    with var_path.open() as handle:
        genes=list(csv.DictReader(handle))
    if len({r['gene'] for r in genes}) != len(genes):
        raise ValueError(f'Duplicate feature genes in {var_path}')
    rows=[]
    for index,row in enumerate(genes[:max_genes or None]):
        gene=row['gene'];ensembl=mapping.get(gene,gene if gene.startswith('ENSG') else '')
        token=tokens.get(ensembl,tokens.get(gene))
        if token is not None and (not isinstance(token,Integral) or isinstance(token,bool) or token<0):
            raise ValueError(f'Invalid token for {gene}: {token}')
        token=int(token) if token is not None else None
        rows.append(dict(gene_index=index,gene=gene,ensembl_id=ensembl,token=token,
                         mapping_status='mapped' if token is not None else 'not_in_token_dictionary'))
    if not rows:raise ValueError('Empty gene catalog')
    return rows


def prepare(output, args):
    if any(x<0 for x in [args.max_genes,args.max_cells,args.train_max_cells]):
        raise ValueError('Caps must be nonnegative; zero means all')
    if min(args.genes_per_shard,args.checkpoint_genes,args.predict_batch_size)<1:
        raise ValueError('Shard/checkpoint/batch sizes must be positive')
    if len(set(args.groups))!=len(args.groups):raise ValueError('Duplicate groups')
    mapping=RUN/'data/reference/gene_symbol_to_ensembl.tsv'
    dictionary=RUN/'data/reference/token_dictionary.pkl'
    checksums={str(p):file_hash(p) for p in [mapping,dictionary]}
    for name in ['all_gene_age.py','all_gene_age_runtime.py','all_gene_age_stats.py','perturb_age.py',
                 'perturb_lineage.py','geneformer_mamba_model.py']:
        checksums[str(SRC/name)]=file_hash(SRC/name)
    group_configs={};panels={}
    for group in args.groups:
        selected_path=RUN/f'diagnostics/{group}_selected_model.json'
        selected=json.loads(selected_path.read_text())
        model=Path(selected['model_path'])
        if not model.is_dir():raise FileNotFoundError(model)
        checksums[str(selected_path)]=file_hash(selected_path)
        var=RUN/f'exported/{group}/var.csv';checksums[str(var)]=file_hash(var)
        train=RUN/f'tokenized/{group}/train.dataset';test=RUN/f'tokenized/{group}/test.dataset'
        genes=catalog(var,mapping,dictionary,args.max_genes);panels[group]=genes
        group_configs[group]=dict(model_path=str(model),selected_epoch=selected['epoch'],
            selected_run_name=selected['run_name'],train_dataset=str(train),test_dataset=str(test),
            train_fingerprint=fingerprint(train),test_fingerprint=fingerprint(test),
            pretrained_fingerprint=fingerprint(model),n_genes=len(genes),
            n_mapped_genes=sum(g['token'] is not None for g in genes),n_shards=math.ceil(len(genes)/args.genes_per_shard))
    config=dict(schema_version=SCHEMA_VERSION,run_dir=str(RUN),groups=list(args.groups),models=group_configs,
        genes_per_shard=args.genes_per_shard,checkpoint_genes=args.checkpoint_genes,
        max_genes=args.max_genes,max_cells=args.max_cells,train_max_cells=args.train_max_cells,
        excluded_evaluation_age_groups=sorted(set(args.exclude_age_group)),
        predict_batch_size=args.predict_batch_size,save_per_cell=args.save_per_cell,
        training=dict(epochs=1.,learning_rate=5e-5,seed=42,batch_size=8,eval_batch_size=16,
                      warmup_steps=0,weight_decay=.01,logging_steps=50),
        perturb_type='delete',label_column='log2_PCW',donor_column='sample_id',
        input_checksums=checksums)
    path=output/'prepared.json'
    if path.exists():
        previous=json.loads(path.read_text())
        for group in previous['models']:previous['models'][group].pop('catalog_sha256',None)
        if previous!=config:
            raise ValueError('Different configuration/inputs already prepared; choose a new --output-dir')
    for group,genes in panels.items():
        write_csv(output/group/'gene_catalog.csv',genes)
        config['models'][group]['catalog_sha256']=file_hash(output/group/'gene_catalog.csv')
    if path.exists() and json.loads(path.read_text())!=config:
        raise ValueError('Prepared gene catalog checksum changed')
    write_json(path,config)
    print(json.dumps({g:{k:v for k,v in c.items() if k in ['n_genes','n_mapped_genes','n_shards']} for g,c in group_configs.items()},indent=2))
    return config


def load_config(output, verify=True):
    config=json.loads((output/'prepared.json').read_text())
    if config['run_dir']!=str(RUN):raise ValueError('Prepared run directory differs from the configured run directory')
    if verify:
        for path,checksum in config['input_checksums'].items():
            if file_hash(path)!=checksum:
                raise ValueError(f'Prepared source/input changed: {path}; use a new output directory')
        for group in config['groups']:
            if file_hash(output/group/'gene_catalog.csv')!=config['models'][group]['catalog_sha256']:
                raise ValueError(f'Gene catalog changed for {group}')
    return config


def read_genes(output,group):
    with (output/group/'gene_catalog.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    for row in rows:
        row['gene_index']=int(row['gene_index'])
        row['token']=int(row['token']) if row['token'] else None
    return rows


def valid_commit(path, config, expected=None):
    if not path.exists():return False
    record=json.loads(path.read_text())
    if record.get('config_sha256')!=config_hash(config):
        raise ValueError(f'Checkpoint configuration mismatch: {path}')
    if expected is not None and record.get('gene_indices')!=expected:
        raise ValueError(f'Checkpoint gene membership mismatch: {path}')
    for name,checksum in record['files'].items():
        target=path.parent/name
        if not target.is_file() or file_hash(target)!=checksum:
            raise ValueError(f'Missing/corrupt checkpoint artifact: {target}')
    return True


def part_specs(config, group, shard):
    count=config['models'][group]['n_genes'];size=config['genes_per_shard']
    if not 0<=shard<config['models'][group]['n_shards']:raise ValueError('Shard index out of range')
    for start in range(shard*size,min((shard+1)*size,count),config['checkpoint_genes']):
        yield start,min(start+config['checkpoint_genes'],(shard+1)*size,count)


def shard_complete(output,config,group,shard):
    return all(valid_commit(output/group/f'parts/{start:06d}/complete.json',config,list(range(start,end)))
               for start,end in part_specs(config,group,shard))




def merge(output,config,groups=None):
    import pandas as pd
    combined=[]
    for group in groups or config['groups']:
        if group not in config['groups']:raise ValueError(f'Group was not prepared: {group}')
        if not valid_commit(output/group/'fit_complete.json',config):raise ValueError(f'Fit incomplete: {group}')
        genes=read_genes(output,group);metrics=[];donor_paths=[];cell_paths=[]
        for shard in range(config['models'][group]['n_shards']):
            if not shard_complete(output,config,group,shard):raise ValueError(f'Incomplete {group} shard {shard}')
            for start,_ in part_specs(config,group,shard):
                part=output/group/f'parts/{start:06d}'
                metrics.append(pd.read_csv(part/'gene_summary.csv'))
                donor_paths.append(part/'donor_level_prediction_deltas.csv.gz')
                if config['save_per_cell']:cell_paths.append(str(part/'per_cell_prediction_deltas.parquet'))
        frame=pd.concat(metrics,ignore_index=True).sort_values('gene_index')
        if frame.gene_index.tolist()!=[g['gene_index'] for g in genes]:raise ValueError(f'Duplicate/missing genes: {group}')
        target=output/group/'gene_summary.csv';temporary=target.with_suffix('.tmp.csv')
        frame.to_csv(temporary,index=False);temporary.replace(target)
        # Stream donor shards rather than retaining all genes x donors in memory.
        target=output/group/'donor_level_prediction_deltas.csv.gz';temporary=target.with_name(target.name+'.tmp')
        import gzip
        with gzip.open(temporary,'wt',newline='') as handle:
            for index,path in enumerate(donor_paths):pd.read_csv(path).to_csv(handle,index=False,header=index==0)
        temporary.replace(target)
        write_json(output/group/'manifest.json',dict(config_sha256=config_hash(config),n_genes=len(frame),
            gene_summary_csv=str(output/group/'gene_summary.csv'),donor_level_prediction_deltas_csv=str(target),
            baseline_cells_parquet=str(output/group/'baseline_cells.parquet'),
            per_cell_partitions=cell_paths,per_cell_semantics='Only target-token-present cells are stored; evaluated genes have zero effect in absent cells; untestable genes have NA effects',
            perturb_type='delete',delta_definition='after minus baseline, log2(PCW)',age_program_assignment=False))
        combined.append(frame)
        print(f'{group}: {len(frame)} genes -> {output/group/"gene_summary.csv"}',flush=True)
    # Publish each model immediately. The combined file waits for all models.
    complete=[]
    for group in config['groups']:
        path=output/group/'manifest.json'
        if not path.exists():break
        record=json.loads(path.read_text())
        if record.get('config_sha256')!=config_hash(config):raise ValueError(f'Merged model configuration mismatch: {group}')
        complete.append(pd.read_csv(output/group/'gene_summary.csv'))
    if len(complete)==len(config['groups']):
        target=output/'all_models_gene_summary.csv'
        temporary=target.with_suffix('.tmp.csv')
        pd.concat(complete,ignore_index=True).to_csv(temporary,index=False)
        temporary.replace(target)




