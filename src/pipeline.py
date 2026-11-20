#!/usr/bin/env python3
"""One explicit entry point for the final DELIGHT-FM workflows."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
from paths import RUN,SRC,TASK
from common import config,inputs,read,digest,source_hashes,settings_hash,atomic_json,immutable,require_plan
GROUPS=['delight_fm','immune','epithelium_colon','epithelium_ileum','stromal']


def plan():
    import training
    settings=config();epochs=settings['sweep']['checkpoint_epochs'];duration=settings['sweep']['training_epochs']
    if not epochs or sorted(set(epochs))!=epochs or min(epochs)<=0 or max(epochs)>duration or any((x*2)%1 for x in epochs):
        raise ValueError('Checkpoint epochs must be unique increasing half-epoch values within training duration')
    if duration!=int(duration) or duration<1:raise ValueError('Training duration must be a positive integer epoch count')
    paths=[p for p in (RUN/'config').glob('*.json') if p.name!='settings.json']+list((RUN/'donors').glob('*.csv'))
    # Large raw inputs are SHA256-verified during materialization and represented
    # by the pinned manifest; avoid rehashing tens of GB for every CLI stage.
    paths+=[p for p in (RUN/'data').rglob('*') if p.is_file() and p.suffix!='.md'
            and not p.is_relative_to(RUN/'data/rna')]
    record=dict(schema_version=1,source_sha256=source_hashes(),settings_sha256=settings_hash(),input_sha256={str(p.relative_to(RUN)):digest(p) for p in sorted(paths)},
        configurations=len(training.conditions()),checkpoint_candidates=len(training.conditions())*len(epochs),
        merged_regulons='CollecTRI only',lineage_regulons='SCENIC+',precision='float32; autocast off; TF32 off')
    record['local_inputs_required']=bool(read(RUN/'config/inputs.json').get('require_local_copies'))
    record['local_inputs_ready']=(RUN/'data/input_manifest.json').exists()
    immutable(RUN/'diagnostics/plan.json',record)
    training.write_table(RUN/'conditions.tsv',training.conditions())
    print(json.dumps({k:v for k,v in record.items() if k not in ['source_sha256','input_sha256']},indent=2))


def task_group(task_id):return GROUPS[task_id]


def submit(kind,dry_run=False):
    import training
    require_plan();slurm=config()['slurm'];path=RUN/'diagnostics'/f'submission_{kind}.json'
    if path.exists() and not dry_run:raise ValueError('Already submitted; inspect '+str(path))
    records=[]
    def job(stage,array=None,gpu=False,after=None):
        cmd=['sbatch','--parsable','--job-name',f'v12-{stage}','--chdir',str(RUN),
            '--time',slurm['time'],
            '--cpus-per-task',str(slurm['cpus']),'--mem',slurm['memory'],
            '--output',str(RUN/'logs'/f'{stage}.%A_%a.out'),'--error',str(RUN/'logs'/f'{stage}.%A_%a.err'),
            '--export','ALL,V12_RUN_DIR='+str(RUN)]
        partition=slurm.get('gpu_partition' if gpu else 'cpu_partition')
        if partition:cmd+=['--partition',partition]
        if gpu:cmd+=['--gpus=1']
        if array:cmd+=['--array',array]
        if after:cmd+=['--dependency','afterok:'+':'.join(after)]
        cmd+=[str(TASK/'runs/job.sh'),stage]
        print(shlex.join(cmd),flush=True)
        ident=f'<{stage}-job-id>' if dry_run else subprocess.check_output(cmd,text=True).strip().split(';')[0]
        records.append(dict(stage=stage,job_id=ident,command=cmd))
        if not dry_run:atomic_json(path,records)
        return ident
    (RUN/'logs').mkdir(exist_ok=True)
    if kind=='training':
        a=job('prepare',f'0-{len(GROUPS)-1}')
        b=job('train',f'0-{len(training.conditions())-1}%{slurm["training_concurrency"]}',True,[a])
        c=job('downstream',f'0-{len(training.conditions())*len(training.EPOCHS)-1}%{slurm["evaluation_concurrency"]}',True,[b])
        d=job('select',f'0-{len(GROUPS)-1}',False,[c])
        e=job('prepare-screens',after=[d])
        job('fit-baseline',f'0-{len(GROUPS)-1}%{slurm["training_concurrency"]}',True,[e])
    elif kind=='perturbations':
        if not dry_run:
            from screens import verify
            for group in GROUPS:verify(group)
        tasks=read(RUN/'diagnostics/perturbation_tasks.json') if (RUN/'diagnostics/perturbation_tasks.json').exists() else None
        # The task counts can be computed before fitting without claiming readiness.
        if tasks is None:
            count=1
            width=config()['perturbation']['tfs_per_shard']
            with (RUN/config()['perturbation']['collectri_table']).open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
            requested=config()['perturbation']['collectri_tfs'];tfs={r['tf'] for r in rows}
            if requested!='all':tfs&=set(requested)
            import math
            count+=math.ceil(len(tfs)/width)
            for group in GROUPS[1:]:
                table='IMMUNE' if group=='immune' else 'STROMAL' if group=='stromal' else 'EPITHELIUM'
                with (RUN/f'data/scenic/{table}.tsv').open() as f:count+=math.ceil(len({r['tf'] for r in csv.DictReader(f,delimiter='\t')})/width)
        else:count=len(tasks)
        a=job('perturb',f'0-{count-1}%{slurm["perturbation_concurrency"]}',True)
        import math
        n=0
        for group in GROUPS[1:]:
            with (RUN/'exported'/group/'var.csv').open() as f:n+=math.ceil((sum(1 for _ in f)-1)/config()['perturbation']['genes_per_shard'])
        b=job('genes',f'0-{n-1}%{slurm["perturbation_concurrency"]}',True)
        c=job('merge',f'0-{len(GROUPS)-1}',False,[a,b])
        job('report',after=[c])
    elif kind=='bhlhe40':
        a=job('bhlhe40-prepare');b=job('bhlhe40',f'0-1%2',True,[a]);job('report',after=[b])


def main():
    stages=['plan','prepare','rebuild-inputs','validate','train','downstream','select','prepare-screens','fit-baseline',
        'perturb','genes','merge','bhlhe40-prepare','bhlhe40','report','cluster','submit-training','submit-perturbations','submit-bhlhe40']
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=stages);p.add_argument('--group',choices=['all']+GROUPS,default='all')
    p.add_argument('--task-id',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID','0')))
    p.add_argument('--direction',choices=['delete','overexpress'])
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    global_stages={'plan','train','downstream','prepare-screens','perturb','genes','report',
                   'submit-training','submit-perturbations','submit-bhlhe40','bhlhe40-prepare','bhlhe40'}
    if args.group!='all' and args.stage in global_stages:
        p.error(f'{args.stage} uses the global manifest/task index; --group is not supported for this stage')
    if args.stage=='plan':plan();return
    if args.stage.startswith('submit-'):submit(args.stage.removeprefix('submit-'),args.dry_run);return
    require_plan()
    if args.stage in ['prepare','validate','rebuild-inputs']:
        import prepare_inputs as data
        groups=GROUPS if args.group=='all' and 'SLURM_ARRAY_TASK_ID' not in os.environ else [args.group if args.group!='all' else task_group(args.task_id)]
        function={'prepare':data.prepare,'validate':data.validate,'rebuild-inputs':data.rebuild}[args.stage]
        for group in groups:function(group)
    elif args.stage in ['train','downstream','select']:
        import training
        if args.stage=='train':training.train(args.task_id)
        elif args.stage=='downstream':training.downstream(args.task_id)
        else:
            groups=GROUPS if args.group=='all' and 'SLURM_ARRAY_TASK_ID' not in os.environ else [args.group if args.group!='all' else task_group(args.task_id)]
            for group in groups:training.selected(group)
    elif args.stage in ['prepare-screens','fit-baseline','perturb','genes','merge']:
        import screens
        if args.stage=='prepare-screens':screens.prepare_screens()
        elif args.stage=='perturb':screens.run_task(args.task_id)
        elif args.stage=='genes':screens.run_genes(args.task_id)
        else:
            groups=GROUPS if args.group=='all' and 'SLURM_ARRAY_TASK_ID' not in os.environ else [args.group if args.group!='all' else task_group(args.task_id)]
            for group in groups:(screens.fit_baseline if args.stage=='fit-baseline' else screens.merge)(group)
    elif args.stage.startswith('bhlhe40'):
        import bhlhe40
        if args.stage=='bhlhe40-prepare':bhlhe40.prepare()
        else:bhlhe40.run_task(args.direction or ['delete','overexpress'][args.task_id])
    elif args.stage=='report':
        import plots
        plots.main()
    elif args.stage=='cluster':
        import cluster_regulons
        groups=GROUPS[1:] if args.group=='all' else [args.group]
        for group in groups:cluster_regulons.run(group)


if __name__=='__main__':main()
