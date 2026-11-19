"""Small shared configuration, hashing and atomic-output helpers."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import sys
from paths import RUN,TASK,SRC


def read(path):
    return json.loads(Path(path).read_text())


def config():
    def expand(value):
        if isinstance(value,str):return value.replace('${WORKSPACE}',str(TASK)).replace('${RUN}',str(RUN))
        if isinstance(value,list):return [expand(x) for x in value]
        if isinstance(value,dict):return {k:expand(v) for k,v in value.items()}
        return value
    result=expand(read(RUN/'config/settings.json'))
    result['model_python']=os.environ.get('PYTHON_BIN',result.get('model_python') or sys.executable)
    return result


def inputs():
    result=read(RUN/'config/inputs.json')
    base=RUN if result.get('path_base')=='run' else TASK
    for record in result['models'].values():
        for key in ['data_root','rds']:
            path=Path(record[key]);record[key]=str(path if path.is_absolute() else base/path)
    return result


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix=path.name+'.',delete=False) as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');tmp=Path(f.name)
    tmp.replace(path)


def immutable(path,value):
    path=Path(path)
    if path.exists() and read(path)!=value:raise ValueError('Immutable record differs: '+str(path))
    if not path.exists():atomic_json(path,value)


def run(command,env=None):
    command=list(map(str,command));print(shlex.join(command),flush=True)
    subprocess.run(command,env=env,cwd=RUN,check=True)


def source_hashes():
    reporting={'plots.py','cluster_regulons.py','figure_style.py','report_assets.py','donor_plotting.py'}
    return {p.name:digest(p) for p in sorted(SRC.iterdir()) if p.suffix in ['.py','.R','.html'] and p.name not in reporting}


def settings_hash():
    settings=read(RUN/'config/settings.json')
    for key in ['slurm','clustering']:settings.pop(key,None)
    return hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()


def require_plan():
    plan=read(RUN/'diagnostics/plan.json')
    if plan['source_sha256']!=source_hashes():raise ValueError('Source changed after planning; create a fresh run/plan')
    if plan['settings_sha256']!=settings_hash():raise ValueError('Computational settings changed after planning; create a fresh run/plan')
    for path,sha in plan['input_sha256'].items():
        if digest(RUN/path)!=sha:raise ValueError('Configuration/reference changed after planning: '+path)
    manifest=RUN/'data/input_manifest.json'
    if read(RUN/'config/inputs.json').get('require_local_copies') and not manifest.exists():
        raise ValueError('Install and verify release inputs with tools/dataset.py before planning')
    if manifest.exists():
        for name,record in read(manifest)['files'].items():
            path=RUN/name
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(RUN):
                raise ValueError('Local input missing or replaced by external link: '+name)
            current=path.stat()
            if current.st_size!=record['bytes'] or current.st_mtime_ns!=record['mtime_ns']:
                raise ValueError('Local input changed after copying: '+name)
    return plan
