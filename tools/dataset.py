#!/usr/bin/env python3
"""Verify, download or upload the checksum-pinned DELIGHT-FM input dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'docs/dataset/manifest.json'

def verify(run, records):
    verified={}
    for name, record in records.items():
        path=run/name
        if not path.resolve().is_relative_to(run.resolve()) or not path.is_file() or path.is_symlink():
            raise ValueError('Missing or externally linked input: '+name)
        stat=path.stat()
        if stat.st_size!=record['bytes']:
            raise ValueError('Size mismatch: '+name)
        h=hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=record['sha256']:
            raise ValueError('SHA256 mismatch: '+name)
        if path.stat().st_mtime_ns!=stat.st_mtime_ns:
            raise ValueError('Input changed during verification: '+name)
        verified[name]={**record,'mtime_ns':stat.st_mtime_ns}
        print('Verified',name,flush=True)
    return verified

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['preview','verify','download','upload'])
    p.add_argument('--run-dir',type=Path,default=(ROOT.parent/'huggingface/dataset' if (ROOT.parent/'huggingface/dataset/data/input_manifest.json').exists() else ROOT/'runs'))
    p.add_argument('--repo-id',default='chloehlee/delight_fm_v1-data',help='Hugging Face dataset namespace/repository')
    p.add_argument('--revision',help='Pinned 40-character commit SHA for downloads')
    p.add_argument('--public',action='store_true',help='Create a public dataset; default private')
    args=p.parse_args();run=args.run_dir.resolve()
    records=json.loads(MANIFEST.read_text())['files']
    if args.action=='preview':
        print(json.dumps({'files':len(records),'GiB':sum(x['bytes'] for x in records.values())/1024**3,'run':str(run)},indent=2));return
    if args.action in ['download','upload'] and not args.repo_id:p.error('--repo-id is required')
    if args.action=='download':
        import re
        if not args.revision or not re.fullmatch('[0-9a-f]{40}',args.revision):p.error('download requires a pinned --revision commit SHA')
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=args.repo_id,repo_type='dataset',revision=args.revision,local_dir=str(run),allow_patterns=list(records))
    verified=verify(run,records)
    if args.action in ['verify','download']:
        target=run/'data/input_manifest.json';target.parent.mkdir(parents=True,exist_ok=True)
        value={'schema_version':1,'files':verified}
        # Preserve an existing valid receipt so repeating verification does not invalidate the run plan.
        if target.exists() and json.loads(target.read_text()).get('files')==verified:return
        if (run/'diagnostics/plan.json').exists():raise ValueError('Receipt changed: start a fresh run/plan before replacing it')
        with tempfile.NamedTemporaryFile('w',dir=target.parent,delete=False) as handle:
            json.dump(value,handle,indent=2);handle.write('\n');tmp=Path(handle.name)
        tmp.replace(target)
    if args.action=='upload':
        from huggingface_hub import HfApi
        api=HfApi();api.create_repo(args.repo_id,repo_type='dataset',private=not args.public,exist_ok=True)
        api.upload_folder(repo_id=args.repo_id,repo_type='dataset',folder_path=str(ROOT/'docs/dataset'),allow_patterns=['README.md','manifest.json'])
        result=api.upload_folder(repo_id=args.repo_id,repo_type='dataset',folder_path=str(run),allow_patterns=list(records),commit_message='Upload checksum-verified DELIGHT-FM v1 inputs')
        print('Dataset revision:',result.oid)

if __name__=='__main__':main()
