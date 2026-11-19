"""Shared saved models, verified baselines and resumable perturbation workers."""
import csv
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from paths import RUN,SRC
from common import config,read,digest,atomic_json,immutable,run,require_plan

LINEAGES=['immune','epithelium_colon','epithelium_ileum','stromal']


def age_root():return RUN/'outputs/perturb_age_all_genes'
def registry(group):return RUN/'outputs/model_registry'/group
def model_path(group):
    if group=='delight_fm':return Path(read(RUN/'diagnostics/delight_fm_selected_model.json')['model_path'])
    return age_root()/group/'age_model'
def reference(group):
    return registry(group)/'classification.npz' if group=='delight_fm' else age_root()/group/'baseline_cells.parquet'


def prepare_screens():
    import all_gene_age as ag
    settings=config()['perturbation']
    for group in ['delight_fm']+LINEAGES:
        if not (RUN/f'diagnostics/{group}_selected_model.json').exists():raise ValueError('Select all model checkpoints first')
    args=SimpleNamespace(groups=LINEAGES,max_genes=0,max_cells=0,train_max_cells=0,
        genes_per_shard=settings['genes_per_shard'],checkpoint_genes=settings['checkpoint_genes'],
        predict_batch_size=settings['batch_size'],save_per_cell=True,exclude_age_group=[])
    if not (age_root()/'prepared.json').exists():
        ag.prepare(age_root(),args)
        cfg=ag.load_config(age_root());cfg['training'].update(config()['age_model'])
        ag.write_json(age_root()/'prepared.json',cfg)
    else:ag.load_config(age_root())
    # Only CollecTRI enters the merged classifier screen.
    source=RUN/settings['collectri_table']
    with source.open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
    if not rows or not {'tf','target','mor'}<=set(rows[0]):raise ValueError('CollecTRI requires tf,target,mor')
    for row in rows:
        if not row['tf'] or not row['target'] or float(row['mor']) not in [-1,1]:raise ValueError('Invalid CollecTRI interaction')
    requested=settings['collectri_tfs'];available={r['tf'] for r in rows}
    tfs=sorted(available if requested=='all' else set(requested)&available)
    if not tfs:raise ValueError('No requested CollecTRI TFs found')
    tasks=[]
    width=settings['tfs_per_shard']
    for start in range(0,len(tfs),width):
        tasks.append(dict(group='delight_fm',kind='regulons',tfs=tfs[start:start+width],
            table=str(source),source='CollecTRI',output=str(RUN/'outputs/delight_fm/regulons'/f'shard_{start//width}')))
    tasks.append(dict(group='delight_fm',kind='markers',genes=sorted(config()['markers']),source='curated cell-type markers',
        output=str(RUN/'outputs/delight_fm/markers/shard_0')))
    for group in LINEAGES:
        table=RUN/'data/scenic'/('IMMUNE.tsv' if group=='immune' else 'STROMAL.tsv' if group=='stromal' else 'EPITHELIUM.tsv')
        with table.open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
        tfs=sorted({r['tf'] for r in rows})
        for start in range(0,len(tfs),width):
            tasks.append(dict(group=group,kind='regulons',tfs=tfs[start:start+width],table=str(table),source='SCENIC+',
                output=str(RUN/'outputs'/group/'regulons'/f'shard_{start//width}')))
    path=RUN/'diagnostics/perturbation_tasks.json'
    immutable(path,tasks)
    print('Prepared',len(tasks),'marker/regulon tasks; no merged SCENIC+ screen.',flush=True)
    return tasks


def binding(group):
    import all_gene_age as ag
    model=model_path(group)
    files=list(model.glob('*.safetensors'))+list(model.glob('pytorch_model*.bin'))
    if not files:raise ValueError('Saved model weights missing: '+str(model))
    return dict(group=group,plan_sha256=digest(RUN/'diagnostics/plan.json'),model=str(model.resolve()),
        model_sha256={q.name:digest(q) for q in files},config_sha256=digest(model/'config.json'),
        datasets={split:ag.fingerprint(RUN/'tokenized'/group/(split+'.dataset')) for split in ['train','test']},
        var_sha256=digest(RUN/'exported'/group/'var.csv'))


def classifier_command():
    return [config()['model_python'],SRC/'perturb_lineage.py','--model-dir',model_path('delight_fm'),
        '--train-dataset',RUN/'tokenized/delight_fm/train.dataset','--test-dataset',RUN/'tokenized/delight_fm/test.dataset',
        '--label-maps',RUN/'tokenized/delight_fm/label_maps.json','--token-dictionary',RUN/'data/reference/token_dictionary.pkl',
        '--gene-mapping-tsv',RUN/'data/reference/gene_symbol_to_ensembl.tsv',
        '--lineage-grouping-csv',RUN/'data/reference/original_lineage_by_cell.csv.gz',
        '--reference-baseline',reference('delight_fm'),'--dataset-split','test','--batch-size',config()['perturbation']['batch_size'],
        '--max-cells','0','--filter-json','{}','--device','cuda','--umap-csv',RUN/'exported/delight_fm/umap_harmony.csv.gz',
        '--umap-plot-top-genes','20','--perturb-type','delete','--output-prefix','v12']


def fit_baseline(group):
    require_plan()
    root=registry(group);root.mkdir(parents=True,exist_ok=True)
    if group=='delight_fm':
        command=classifier_command()+['--baseline-only','--output-dir',root]
        run(command)
    else:
        import all_gene_age as ag
        from all_gene_age_runtime import fit
        fit(age_root(),ag.load_config(age_root()),group)
    immutable(root/'binding.json',binding(group))
    # Model-visible input identity checks are mandatory before any deletion shards.
    from controls import run_controls
    controls=run_controls(group)
    files=[reference(group)]
    if group!='delight_fm':files.append(reference(group).parent/'baseline_embeddings.npy')
    immutable(root/'baseline_verified.json',dict(binding_sha256=digest(root/'binding.json'),
        files={str(q):digest(q) for q in files},controls_sha256=digest(controls),
        inference_precision='float32; autocast off; TF32 off'))
    print(group+': one saved model and baseline ready',flush=True)


def verify(group):
    require_plan();root=registry(group)
    if read(root/'binding.json')!=binding(group):raise ValueError('Saved model or datasets changed: '+group)
    record=read(root/'baseline_verified.json')
    if record['binding_sha256']!=digest(root/'binding.json'):raise ValueError('Baseline binding mismatch')
    for path,sha in record['files'].items():
        if digest(path)!=sha:raise ValueError('Baseline file changed: '+path)
    if record['controls_sha256']!=digest(root/'noop_controls.json'):raise ValueError('Control record changed')
    return record


def age_command(group):
    return [config()['model_python'],SRC/'perturb_age.py','--model-dir',model_path(group),'--skip-training',
        '--reference-baseline',reference(group),'--train-dataset',RUN/'tokenized'/group/'train.dataset',
        '--test-dataset',RUN/'tokenized'/group/'test.dataset','--token-dictionary',RUN/'data/reference/token_dictionary.pkl',
        '--gene-mapping-tsv',RUN/'data/reference/gene_symbol_to_ensembl.tsv','--label-column','log2_PCW',
        '--donor-column','sample_id','--filter-json','{}','--train-filter-json','{}','--perturb-dataset-split','test',
        '--predict-batch-size',config()['perturbation']['batch_size'],'--max-cells','0','--device','cuda',
        '--umap-csv',RUN/'exported'/group/'umap_harmony.csv.gz','--output-prefix','v12']


def task_command(task):
    group=task['group'];command=classifier_command() if group=='delight_fm' else age_command(group)
    if task['kind']=='markers':
        command+=['--genes',','.join(task['genes']),'--marker-lineage-map-json',json.dumps(config()['markers'])]
    else:
        command+=['--regulon-tsv',task['table'],'--regulon-tfs',','.join(task['tfs']),
            '--regulon-max-targets','0','--regulon-include-tf','--regulon-signed-mode','all']
        if group=='delight_fm':command+=['--marker-lineage-map-json','none']
        else:command+=['--only-regulons','--adult-genes','','--fetal-genes','','--early-life-genes','']
    return command+['--output-dir',task['output']]


def committed(folder,group):
    path=folder/'complete.json'
    if not path.exists():return False
    record=read(path)
    if record['binding_sha256']!=digest(registry(group)/'binding.json'):raise ValueError('Shard model binding mismatch')
    for name,sha in record['files'].items():
        if digest(folder/name)!=sha:raise ValueError('Committed result changed: '+name)
    return True


def run_task(task_id):
    task=read(RUN/'diagnostics/perturbation_tasks.json')[task_id]
    group=task['group'];verify(group);folder=Path(task['output'])
    if committed(folder,group):return
    run(task_command(task))
    manifest=read(folder/'v12_delete_manifest.json')
    if not manifest['baseline_verified'] or not manifest['training_skipped']:raise ValueError('Perturbation model was not reused')
    files={str(q.relative_to(folder)):digest(q) for q in folder.rglob('*') if q.is_file() and q.name!='complete.json'}
    atomic_json(folder/'complete.json',dict(task=task,binding_sha256=digest(registry(group)/'binding.json'),files=files))


def run_genes(task_id):
    import all_gene_age as ag
    from all_gene_age_runtime import run_shard
    cfg=ag.load_config(age_root());tasks=[(g,i) for g in LINEAGES for i in range(cfg['models'][g]['n_shards'])]
    group,shard=tasks[task_id];verify(group)
    run_shard(age_root(),cfg,group,shard)


def merge(group):
    import pandas as pd
    verify(group)
    tasks=[x for x in read(RUN/'diagnostics/perturbation_tasks.json') if x['group']==group]
    for kind in sorted({x['kind'] for x in tasks}):
        tables={};count=0
        for task in [x for x in tasks if x['kind']==kind]:
            folder=Path(task['output'])
            if not committed(folder,group):raise ValueError('Incomplete perturbation task: '+str(folder))
            for name in (['summary','lineage_scores','lineage_group_scores'] if group=='delight_fm'
                         else ['summary','donor_level_prediction_deltas']):
                tables.setdefault(name,[]).append(pd.read_csv(folder/f'v12_delete_{name}.csv'))
        dest=RUN/'outputs'/group/kind;dest.mkdir(parents=True,exist_ok=True)
        for name,frames in tables.items():
            result=pd.concat(frames,ignore_index=True)
            if name=='summary' and result.perturbation_label.duplicated().any():raise ValueError('Duplicate perturbations')
            result.to_csv(dest/f'v12_delete_{name}.csv',index=False)
    if group!='delight_fm':
        import all_gene_age as ag
        ag.merge(age_root(),ag.load_config(age_root()),[group])
    atomic_json(RUN/'outputs'/group/'complete.json',dict(binding_sha256=digest(registry(group)/'binding.json'),status='complete'))
