"""Proportional BHLHE40 overexpression and inverse-response deletion only."""
import csv
import json
from pathlib import Path
from paths import RUN
from common import config,atomic_json,immutable,digest,read,run
from signed_tf_targets import resolve_targets,make_programs
from screens import verify,age_command,model_path,reference,registry,committed


def prepare():
    settings=config()['bhlhe40'];group=settings['group'];verify(group)
    root=RUN/'outputs'/group/'bhlhe40'
    source=RUN/settings['target_table']
    strengths=settings['strengths']
    for direction in ['delete','overexpress']:
        out=root/direction;out.mkdir(parents=True,exist_ok=True)
        plan,coverage=resolve_targets(source,RUN/'exported'/group/'var.csv',RUN/'data/reference/gene_symbol_to_ensembl.tsv',
            RUN/'data/reference/token_dictionary.pkl',tf=settings['tf'],perturb_type=direction,
            max_length=config()['sweep']['max_length'])
        if plan['signed_edit_version']!='float32_v2':raise ValueError('Float32-stable editor required')
        immutable(out/'signed_target_plan.json',plan)
        with (out/'target_coverage.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(coverage[0]));writer.writeheader();writer.writerows(coverage)
        immutable(out/'prepared.json',dict(group=group,direction=direction,strengths=strengths,
            model=str(model_path(group)),reference_baseline=str(reference(group)),
            binding_sha256=digest(registry(group)/'binding.json'),source_sha256=digest(source),
            plan_sha256=digest(out/'signed_target_plan.json'),programs=[p['perturbation_label'] for p in make_programs(plan,strengths)],
            interpretation='Deletion imposes the inverse bulk overexpression signature; it is not a measured knockout response.'))
    print('Prepared proportional deletion and overexpression; three strengths plus TF-only and targets-only controls.',flush=True)


def command(direction):
    settings=config()['bhlhe40'];group=settings['group'];out=RUN/'outputs'/group/'bhlhe40'/direction
    prepared=read(out/'prepared.json')
    if prepared['binding_sha256']!=digest(registry(group)/'binding.json'):raise ValueError('BHLHE40 model binding changed')
    if prepared['plan_sha256']!=digest(out/'signed_target_plan.json'):raise ValueError('BHLHE40 plan changed')
    cmd=age_command(group)
    i=cmd.index('--output-prefix');cmd[i+1]='bhlhe40'
    return cmd+['--signed-target-plan',out/'signed_target_plan.json',
        '--signed-target-strengths',','.join(map(str,prepared['strengths'])),'--perturb-type',direction,'--output-dir',out]


def run_task(direction):
    group=config()['bhlhe40']['group'];verify(group)
    out=RUN/'outputs'/group/'bhlhe40'/direction
    if committed(out,group):return
    run(command(direction))
    manifest=read(out/f'bhlhe40_{direction}_manifest.json')
    if not manifest['training_skipped'] or not manifest['baseline_verified']:raise ValueError('BHLHE40 must reuse model and reference')
    atomic_json(out/'complete.json',dict(binding_sha256=digest(registry(group)/'binding.json'),
        files={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='complete.json'}))
