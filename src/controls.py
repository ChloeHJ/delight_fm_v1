"""Mandatory no-op checks on actual held-out cells with the saved model."""
from paths import RUN
def run_controls(group):
    from screens import model_path,registry
    from common import config,atomic_json,digest
    p=dict(group=group,model=str(model_path(group)),task="classification" if group=="delight_fm" else "age_regression",
        test_dataset=str(RUN/"tokenized"/group/"test.dataset"),batch_size=config()["perturbation"]["batch_size"],
        plan_id=digest(RUN/"diagnostics/plan.json"))
    """Exercise the corrected editor with the actual saved model, before shards."""
    import numpy as np
    import torch
    from datasets import load_from_disk
    from baseline_contract import strict_classifier,strict_age_model,compare_arrays
    from signed_tf_targets import edit_signed_rows
    from perturb_lineage import PadCollator,pooled_backbone
    from perturb_age import RegressionCollator,predict_and_embed_rows
    ds=load_from_disk(p['test_dataset'])
    per_donor={};indices=[]
    for i,donor in enumerate(ds['sample_id']):
        if per_donor.get(donor,0)<8:
            indices.append(i);per_donor[donor]=per_donor.get(donor,0)+1
        if len(indices)>=104:break
    model=(strict_classifier(p['model']) if p['task']=='classification' else strict_age_model(p['model'])).eval().to('cuda')
    maximum=int(model.config.max_position_embeddings)
    from all_gene_age_runtime import retained_rows
    rows=retained_rows(ds,indices,maximum)
    tokens=sorted({int(t) for row in rows for t in row['input_ids']})[:16]
    absent=int(model.config.vocab_size)+1
    base=dict(version=2,signed_edit_version='float32_v2',tf='CONTROL',tf_token=absent,tf_edit=False,
        perturb_type='delete',response_sign=-1,value_clip=1e9,max_length=maximum,
        targets=[dict(token=t,bulk_log2FC=1.,mean=100.,sd=1.) for t in tokens])
    cases=[('zero_strength',base,0.),
           ('zero_fold_change',dict(base,targets=[dict(t,bulk_log2FC=0.) for t in base['targets']]),1.),
           ('absent_targets',dict(base,targets=[dict(token=absent,bulk_log2FC=1.,mean=100.,sd=1.)]),1.)]
    cls=p['task']=='classification'
    collator=(PadCollator if cls else RegressionCollator)(int(model.config.pad_token_id),maximum)
    results=[]
    with torch.inference_mode():
        for name,edit_plan,strength in cases:
            edited,_=edit_signed_rows(rows,edit_plan,strength)
            for start in range(0,len(rows),p['batch_size']):
                original_batch=collator(rows[start:start+p['batch_size']]);edited_batch=collator(edited[start:start+p['batch_size']])
                for key in original_batch:
                    if not torch.equal(original_batch[key],edited_batch[key]):
                        raise ValueError('No-op changed model input: '+name+'/'+key)
                if cls:
                    before=pooled_backbone(model,original_batch,torch.device('cuda'))
                    after=pooled_backbone(model,edited_batch,torch.device('cuda'))
                else:
                    before=predict_and_embed_rows(model,rows[start:start+p['batch_size']],collator,torch.device('cuda'))
                    after=predict_and_embed_rows(model,edited[start:start+p['batch_size']],collator,torch.device('cuda'))
                for b,a in zip(before,after):
                    if b is not None:
                        compare_arrays(np.asarray(a),np.asarray(b),name,atol=1e-7)
            results.append(name)
    path=registry(group)/'noop_controls.json'
    atomic_json(path,dict(plan_id=p['plan_id'],n_cells=len(rows),passed=results,
        signed_edit_version='float32_v2',binding_sha256=digest(registry(group)/'binding.json')))
    return path
