"""GPU fit/inference workers for the resumable all-gene deletion screen."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import Trainer, set_seed

from all_gene_age import (config_hash,file_hash,fingerprint,write_json,read_genes,
                          valid_commit,part_specs,shard_complete)
from all_gene_age_stats import build_token_postings,token_cells,summarize_gene
from geneformer_mamba_model import GeneformerMambaForSequenceClassification
from perturb_age import (RegressionCollator,add_numeric_labels,build_training_args,
                         compute_regression_metrics,metadata_frame,predict_and_embed_rows)
from perturb_lineage import cap_dataset,perturb_rows


def require_gpu():
    if not torch.cuda.is_available():raise RuntimeError('Fit/inference requires a CUDA GPU; submit via workflow_all_gene_age.sh')


def load_age_model(path):
    from baseline_contract import strict_age_model
    return strict_age_model(path)


def evaluation_dataset(config,group):
    dataset=add_numeric_labels(load_from_disk(config['models'][group]['test_dataset']),'log2_PCW')
    excluded=set(config['excluded_evaluation_age_groups'])
    if excluded:
        if 'AgeGroup' not in dataset.column_names:raise ValueError('AgeGroup missing for requested exclusion')
        dataset=dataset.filter(lambda row: row['AgeGroup'] not in excluded,keep_in_memory=True)
    dataset=cap_dataset(dataset,config['max_cells'],config['training']['seed'])
    if not len(dataset):raise ValueError('No evaluation cells remain')
    return dataset


def retained_rows(dataset,indices,max_length):
    rows=[]
    for index in indices:
        row=dict(dataset[int(index)])
        for column in ['expression_values','auxiliary_gene_mask']:
            if column in row and len(row[column])!=len(row['input_ids']):
                raise ValueError(f'{column} does not align with input_ids')
        row['input_ids']=row['input_ids'][:max_length]
        for column in ['expression_values','auxiliary_gene_mask']:
            if column in row:row[column]=row[column][:max_length]
        row['length']=len(row['input_ids'])
        rows.append(row)
    return rows


def commit(directory,config,files,**extra):
    write_json(directory/'complete.json',dict(config_sha256=config_hash(config),
               files={name:file_hash(directory/name) for name in files},**extra))


def fit(output,config,group):
    from baseline_contract import configure_precision
    configure_precision()
    require_gpu();directory=output/group
    if valid_commit(directory/'fit_complete.json',config):
        print(f'{group}: fixed age model/baseline already complete');return
    model_config=config['models'][group]
    for key,name in [('train_dataset','train_fingerprint'),('test_dataset','test_fingerprint'),('model_path','pretrained_fingerprint')]:
        if fingerprint(model_config[key])!=model_config[name]:raise ValueError(f'Input changed: {model_config[key]}')
    settings=config['training'];set_seed(settings['seed'])
    train=add_numeric_labels(load_from_disk(model_config['train_dataset']),'log2_PCW')
    test=add_numeric_labels(load_from_disk(model_config['test_dataset']),'log2_PCW')
    train=cap_dataset(train,config['train_max_cells'],settings['seed'])
    if not len(train) or not len(test):raise ValueError('Empty training/evaluation dataset')
    model=GeneformerMambaForSequenceClassification.from_pretrained(model_config['model_path'],
           num_labels=1,problem_type='regression',ignore_mismatched_sizes=True)
    max_length=int(model.config.max_position_embeddings)
    collator=RegressionCollator(int(model.config.pad_token_id),max_length)
    args=SimpleNamespace(downstream_epochs=settings['epochs'],learning_rate=settings['learning_rate'],
        batch_size=settings['batch_size'],eval_batch_size=settings['eval_batch_size'],
        warmup_steps=settings['warmup_steps'],weight_decay=settings['weight_decay'],
        logging_steps=settings['logging_steps'],seed=settings['seed'])
    trainer=Trainer(model=model,args=build_training_args(args,directory/'regression_training'),
                    train_dataset=train,eval_dataset=test,data_collator=collator,compute_metrics=compute_regression_metrics)
    training=trainer.train();metrics=trainer.evaluate()
    trainer.save_model(str(directory/'age_model'))
    dataset=evaluation_dataset(config,group)
    model.eval().to('cuda')
    predictions=[];embeddings=[];batch_size=config['predict_batch_size']
    for start in range(0,len(dataset),batch_size):
        rows=retained_rows(dataset,range(start,min(start+batch_size,len(dataset))),max_length)
        pred,embedding=predict_and_embed_rows(model,rows,collator,torch.device('cuda'))
        predictions.append(pred);embeddings.append(embedding.numpy())
    predictions=np.concatenate(predictions);embeddings=np.concatenate(embeddings)
    if not np.isfinite(predictions).all() or not np.isfinite(embeddings).all():raise ValueError('Nonfinite baseline')
    meta=metadata_frame(dataset,'log2_PCW','sample_id').reset_index(drop=True)
    if 'cell_id' not in meta:raise ValueError('cell_id is required for reusable cell-level outputs')
    if meta.cell_id.duplicated().any() or meta.sample_id.isna().any():raise ValueError('Invalid cell/donor identifiers')
    meta.insert(0,'cell_index',np.arange(len(meta),dtype=np.int32))
    meta['base_pred_value']=predictions
    meta.to_parquet(directory/'baseline_cells.parquet',index=False,compression='zstd')
    np.save(directory/'baseline_embeddings.npy',embeddings)
    postings=build_token_postings((dataset[i] for i in range(len(dataset))),max_length,int(model.config.pad_token_id))
    np.savez_compressed(directory/'token_postings.npz',**postings)
    genes=read_genes(output,group)
    coverage=[dict(g,n_cells=len(meta),present_cells=len(token_cells(postings,g['token'])) if g['token'] is not None else 0) for g in genes]
    pd.DataFrame(coverage).to_csv(directory/'gene_coverage.csv',index=False)
    info=dict(config_sha256=config_hash(config),group=group,n_train_cells=len(train),n_eval_cells=len(test),
        n_perturb_cells=len(dataset),n_donors=meta.sample_id.nunique(),max_length=max_length,
        n_target_cell_pairs=sum(row['present_cells'] for row in coverage),
        training_loss=float(training.training_loss),eval_metrics={k:float(v) for k,v in metrics.items() if isinstance(v,(float,int))},
        dataset_fingerprint=dataset._fingerprint)
    write_json(directory/'fit_info.json',info)
    files=['baseline_cells.parquet','baseline_embeddings.npy','token_postings.npz','gene_coverage.csv','fit_info.json']
    files += [str(p.relative_to(directory)) for p in (directory/'age_model').glob('*') if p.is_file()]
    write_json(directory/'fit_complete.json',dict(config_sha256=config_hash(config),
               files={name:file_hash(directory/name) for name in files}))
    print(json.dumps(info,indent=2),flush=True)


def run_shard(output,config,group,shard):
    require_gpu();directory=output/group
    if shard_complete(output,config,group,shard):print(f'{group} shard {shard}: complete');return
    if not valid_commit(directory/'fit_complete.json',config):raise ValueError('Fixed age model/baseline incomplete')
    if fingerprint(config['models'][group]['test_dataset'])!=config['models'][group]['test_fingerprint']:
        raise ValueError('Test input changed')
    model=load_age_model(directory/'age_model')
    model.eval().to('cuda');max_length=int(model.config.max_position_embeddings)
    collator=RegressionCollator(int(model.config.pad_token_id),max_length)
    dataset=evaluation_dataset(config,group)
    meta=pd.read_parquet(directory/'baseline_cells.parquet')
    if meta.cell_id.astype(str).tolist()!=list(map(str,dataset['cell_id'])):raise ValueError('Baseline cell order differs')
    baseline=meta.base_pred_value.to_numpy(dtype=float)
    base_embeddings=np.load(directory/'baseline_embeddings.npy',mmap_mode='r')
    postings=np.load(directory/'token_postings.npz')
    genes=read_genes(output,group);batch_size=config['predict_batch_size']
    for start,end in part_specs(config,group,shard):
        part=directory/f'parts/{start:06d}'
        if valid_commit(part/'complete.json',config,list(range(start,end))):continue
        part.mkdir(parents=True,exist_ok=True)
        summaries=[];donor_frames=[];cell_frames=[]
        for gene in genes[start:end]:
            indices=token_cells(postings,gene['token']) if gene['token'] is not None else np.empty(0,dtype=np.int64)
            predictions=[];l2=[];cosine=[]
            for offset in range(0,len(indices),batch_size):
                batch_indices=indices[offset:offset+batch_size]
                rows=retained_rows(dataset,batch_indices,max_length)
                edited=perturb_rows(rows,[gene['token']],'delete',None)
                pred,embedding=predict_and_embed_rows(model,edited,collator,torch.device('cuda'))
                original=torch.from_numpy(np.array(base_embeddings[batch_indices],copy=True))
                predictions.append(pred)
                l2.append(torch.linalg.vector_norm(embedding-original,dim=1).numpy())
                cosine.append(F.cosine_similarity(embedding,original,dim=1).numpy())
            concatenate=lambda chunks:np.concatenate(chunks) if chunks else np.empty(0,dtype=float)
            summary,donors,cells=summarize_gene(dict(gene,group=group),meta,baseline,indices,
                                               concatenate(predictions),concatenate(l2),concatenate(cosine))
            summaries.append(summary);donor_frames.append(donors)
            if config['save_per_cell']:cell_frames.append(cells)
            print(f'{group} {gene["gene"]}: {len(indices)}/{len(meta)} cells, delta={summary["delta_pred_mean"]}',flush=True)
        pd.DataFrame(summaries).to_csv(part/'gene_summary.csv',index=False)
        pd.concat(donor_frames,ignore_index=True).to_csv(part/'donor_level_prediction_deltas.csv.gz',index=False,compression='gzip')
        files=['gene_summary.csv','donor_level_prediction_deltas.csv.gz']
        if config['save_per_cell']:
            pd.concat(cell_frames,ignore_index=True).to_parquet(part/'per_cell_prediction_deltas.parquet',index=False,compression='zstd')
            files.append('per_cell_prediction_deltas.parquet')
        commit(part,config,files,gene_indices=list(range(start,end)))
    print(f'{group} shard {shard}: complete',flush=True)
