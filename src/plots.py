"""Final loss/task, perturbation, UMAP and proportional BHLHE40 figures."""
import html
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import RUN
from common import config,read,digest,atomic_json
from figure_style import apply,rank_figure
from donor_plotting import donor_grid
from lineage_groups import grouped_cell_effects,correlation_linkages
from scipy.cluster.hierarchy import leaves_list
from report_assets import embed_images
apply()


def save(fig,out,stem,title):
    (out/'figures').mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:fig.savefig(out/'figures'/f'{stem}.{ext}',dpi=150,bbox_inches='tight')
    plt.close(fig)
    return f'<figure><h3>{html.escape(title)}</h3><img src="figures/{stem}.png"><figcaption><a href="figures/{stem}.pdf">PDF</a> · <a href="figures/{stem}.png">PNG</a></figcaption></figure>'


def table(frame):
    return '<div class="table">'+frame.to_html(index=False,border=0,float_format=lambda x:f'{x:.5g}',na_rep='—')+'</div>'


def page(out,title,body):
    content='<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    content+='<title>'+html.escape(title)+'</title><style>body{font:15px/1.5 Arial,sans-serif;max-width:1400px;margin:auto;padding:24px;background:#f4f6f9;color:#172033}section{background:white;padding:22px;margin:20px 0;border:1px solid #dce3eb;border-radius:10px}.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,560px),1fr));gap:18px}figure{margin:12px 0}img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:13px;width:100%}.table{max-height:600px;overflow:auto}td,th{padding:8px;text-align:left;border-bottom:1px solid #dce3eb}a{color:#176d78}figcaption{font-size:12px}summary{cursor:pointer}</style></head><body>'
    content+='<h1>'+html.escape(title)+'</h1>'+body+'</body></html>'
    out.mkdir(parents=True,exist_ok=True);(out/'index.html').write_text(embed_images(content,out))


def heatmap(matrix,out,stem,title,signed=True):
    matrix.to_csv(out/(stem+'_values.csv'))
    values=matrix.to_numpy(float)
    fig,ax=plt.subplots(figsize=(.75*max(8,.55*len(matrix.columns)+2),.75*max(3.5,.3*len(matrix)+2)),layout='constrained')
    bound=max(float(np.nanmax(np.abs(values))),1e-9) if np.isfinite(values).any() else 1
    image=ax.imshow(np.ma.masked_invalid(values),aspect='auto',cmap='coolwarm' if signed else 'viridis',
        vmin=-bound if signed else None,vmax=bound if signed else None)
    ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=70,ha='right',fontsize=8)
    ax.set_yticks(range(len(matrix)),matrix.index,fontsize=8)
    ax.set_title(title);fig.colorbar(image,ax=ax)
    return save(fig,out,stem,title)


def training_report():
    import training
    out=RUN/'outputs/web_summary/training';out.mkdir(parents=True,exist_ok=True)
    sections=[]
    for group in training.GROUPS:
        history=[];metrics=[]
        for row in [r for r in training.conditions() if r['group']==group]:
            path=RUN/'pretrain_outputs'/row['run_name']/training.FINAL_EPOCH_LABEL/'metrics/trainer_log_history.csv'
            if path.exists():
                frame=pd.read_csv(path);frame['run_name']=row['run_name'];history.append(frame)
            for path in (RUN/'outputs/downstream'/row['run_name']).glob('epoch_*/*_summary.csv'):
                frame=pd.read_csv(path)
                if 'task' not in frame:continue
                frame['run_name']=row['run_name'];frame['pretrain_epoch']=float(path.parent.name.removeprefix('epoch_').replace('p','.'))
                metrics.append(frame)
        plots=[]
        if history:
            data=pd.concat(history,ignore_index=True);data.to_csv(out/f'{group}_pretraining_history.csv',index=False)
            fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
            for name,part in data.groupby('run_name'):
                for ax,key in zip(axes,['loss','eval_loss']):
                    p=part.dropna(subset=[key,'epoch']).sort_values('epoch');ax.plot(p.epoch,p[key],label=name.removeprefix(group+'_'))
            for ax,title in zip(axes,['Training loss','Validation loss']):
                ax.set(xlabel='Pretraining epoch',ylabel=title,title=title);ax.legend(fontsize=7);ax.grid(alpha=.2)
            plots.append(save(fig,out,group+'_loss','Training and validation loss'))
        if metrics:
            data=pd.concat(metrics,ignore_index=True);data.to_csv(out/f'{group}_downstream_metrics.csv',index=False)
            for metric in ['eval_macro_f1','eval_accuracy','eval_pearson']:
                tasks=['log2_pcw_regression'] if metric=='eval_pearson' else ['broadcluster','cluster','location']
                fig,axes=plt.subplots(1,len(tasks),figsize=(5*len(tasks),3.6),squeeze=False,layout='constrained')
                for ax,task in zip(axes.flat,tasks):
                    part=data[data.task.eq(task)].dropna(subset=[metric])
                    for name,p in part.groupby('run_name'):
                        p=p.sort_values('pretrain_epoch');ax.plot(p.pretrain_epoch,p[metric],'.-',label=name.removeprefix(group+'_'),ms=3)
                    ax.set(title=task,xlabel='Pretraining checkpoint epoch',ylabel=metric.removeprefix('eval_'));ax.grid(alpha=.2)
                    if len(part):ax.legend(fontsize=7)
                    else:ax.text(.5,.5,'Not estimable / no output',ha='center',transform=ax.transAxes)
                plots.append(save(fig,out,group+'_'+metric,'Downstream '+metric.removeprefix('eval_')))
            for task,metric in [('cluster','eval_macro_f1'),('broadcluster','eval_macro_f1'),('log2_pcw_regression','eval_pearson')]:
                part=data[data.task.eq(task)].dropna(subset=[metric])
                if len(part):
                    matrix=part.pivot(index='run_name',columns='pretrain_epoch',values=metric).sort_index(axis=1)
                    matrix.index=matrix.index.str.replace(group+'_','',regex=False)
                    plots.append(heatmap(matrix,out,group+'_'+task+'_sweep',task+' · '+metric,False))
            age=data[data.task.eq('log2_pcw_regression')]
            if 'cv_mean_pearson' in age and age.cv_mean_pearson.notna().any():
                fig,ax=plt.subplots(figsize=(6,3.6),layout='constrained')
                for name,part in age.groupby('run_name'):
                    part=part.sort_values('pretrain_epoch');ax.plot(part.pretrain_epoch,part.cv_mean_pearson,'.-',label=name.removeprefix(group+'_'))
                ax.set(xlabel='Pretraining checkpoint epoch',ylabel='Mean donor-grouped CV Pearson',title='Fivefold training-donor cross-validation');ax.legend(fontsize=7)
                plots.append(save(fig,out,group+'_age_CV','Age regression · donor-grouped training CV'))
            rankpath=RUN/'outputs/model_selection'/f'{group}_ranking.json'
            if rankpath.exists():
                rank=pd.DataFrame(read(rankpath));matrix=rank.pivot(index='run_name',columns='epoch',values='selection_score').sort_index(axis=1)
                matrix.index=matrix.index.str.replace(group+'_','',regex=False)
                plots.append(heatmap(matrix,out,group+'_selection_score','Checkpoint selection score',False))
            selected=RUN/'diagnostics'/f'{group}_selected_model.json'
            if selected.exists():
                choice=read(selected);chosen=data[data.run_name.eq(choice['run_name'])&data.pretrain_epoch.eq(choice['epoch'])]
                for row in chosen.itertuples():
                    directory=Path(row.output_dir)
                    if row.task=='log2_pcw_regression':continue
                    path=directory/'confusion_matrix_row_normalized.csv'
                    if path.exists():plots.append(heatmap(pd.read_csv(path,index_col=0),out,group+'_'+row.task+'_confusion','Selected '+row.task+' · row-normalized confusion',False))
                chosen.to_csv(out/f'{group}_selected_downstream_metrics.csv',index=False)
        sections.append('<section><h2>'+group+'</h2><div class="plots">'+''.join(plots)+'</div></section>')
    page(out,'Training and downstream sweeps','<p>Classification shows macro F1 and accuracy; age regression shows evaluation Pearson. Donor-grouped CV metrics are retained separately in the downloadable tables. The evaluation subset is also used for checkpoint selection.</p>'+''.join(sections))
    return out/'index.html'


def age_report(group):
    root=RUN/'outputs'/group;out=root/'web_summary';out.mkdir(parents=True,exist_ok=True)
    basepath=RUN/'outputs/perturb_age_all_genes'/group/'baseline_cells.parquet'
    if not basepath.exists():return None
    base=pd.read_parquet(basepath);base['sample_id']=base.sample_id.astype(str)
    meta=base.groupby('sample_id').agg(true_mean=('true_value','mean'),predicted_mean=('base_pred_value','mean'),n_cells=('true_value','size'),AgeGroup=('AgeGroup','first')).reset_index().sort_values('true_mean')
    meta.to_csv(out/'baseline_donor_predictions.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,4.5),layout='constrained')
    for age,p in meta.groupby('AgeGroup',sort=False):ax.scatter(p.true_mean,p.predicted_mean,s=15+np.sqrt(p.n_cells)*1.5,label=age,alpha=.8)
    lo=min(meta.true_mean.min(),meta.predicted_mean.min());hi=max(meta.true_mean.max(),meta.predicted_mean.max())
    ax.plot([lo,hi],[lo,hi],'--',color='grey');ax.set(xlabel='Donor true mean log2(PCW)',ylabel='Donor mean predicted log2(PCW)',title='Shared saved age-model baseline')
    ax.legend(fontsize=7,bbox_to_anchor=(1,1),loc='upper left')
    pictures=[save(fig,out,'baseline_age','Final saved age model · donor means')]
    summarypath=root/'regulons/v12_delete_summary.csv';donorpath=root/'regulons/v12_delete_donor_level_prediction_deltas.csv'
    if summarypath.exists() and donorpath.exists() and (root/'complete.json').exists():
        summary=pd.read_csv(summarypath);donors=pd.read_csv(donorpath,dtype={'sample_id':str})
        if donors.duplicated(['perturbation_label','sample_id']).any():raise ValueError('Duplicate donor response')
        equal=donors.groupby('perturbation_label').delta_pred_mean.mean().rename('equal_donor_delta')
        summary=summary.merge(equal,on='perturbation_label',validate='one_to_one')
        summary=summary.reindex(summary.equal_donor_delta.abs().sort_values(ascending=False).index)
        summary.to_csv(out/'regulons_ranked.csv',index=False)
        early=donors[donors.AgeGroup.isin(['Newborn','< 1 Month','< 2 Months','< 6 Months','< 1 Year','< 2 Years'])]
        early.groupby('perturbation_label').delta_pred_mean.agg(lambda x:x.abs().mean()).rename('mean_absolute_early_life_effect').sort_values(ascending=False).to_csv(out/'early_life_magnitude_ranking.csv')
        pictures.append(save(rank_figure(summary,'perturbation_label','equal_donor_delta','Regulon deletions · equal donor weights'),out,'regulon_ranking','Ranked regulon age changes'))
        names=summary.head(20).perturbation_label.tolist()
        matrix=donors[donors.perturbation_label.isin(names)].pivot(index='perturbation_label',columns='sample_id',values='delta_pred_mean').reindex(index=names,columns=meta.sample_id)
        matrix.columns=[f'{r.AgeGroup} | {r.sample_id}' for r in meta.itertuples()]
        pictures.append(heatmap(matrix,out,'regulon_donor_heatmap','Top regulons · donor prediction changes'))
        focus=[x for x in summary.perturbation_label if x.split('_')[0] in config()['clustering']['focus_tfs']]
        for stem,labels in [('top_regulons',names[:12]),('predefined_regulons',focus)]:
            if not labels:continue
            fig,values=donor_grid(donors,'delta_pred_mean',labels,group+' · donor age changes',columns=3,shared_y=True)
            values.to_csv(out/f'{stem}_donor_values.csv',index=False);pictures.append(save(fig,out,stem+'_donor_dots','Regulon donor responses'))
        # Baseline and perturbed predicted-age means remain in raw regression units.
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        ax.plot(meta.true_mean,meta.predicted_mean,'o-',color='grey',label='Baseline')
        for name in names[:4]:
            part=donors[donors.perturbation_label.eq(name)].set_index('sample_id').reindex(meta.sample_id)
            ax.plot(part.true_mean,part.perturbed_pred_mean,'.-',label=name,lw=1)
        ax.set(xlabel='Donor true mean log2(PCW)',ylabel='Donor mean predicted log2(PCW)',title='Baseline and top regulon deletions');ax.legend(fontsize=7)
        pictures.append(save(fig,out,'regulon_age_progression','Baseline and perturbed donor mean ages'))
    genespath=RUN/'outputs/perturb_age_all_genes'/group/'gene_summary.csv'
    if genespath.exists():
        genes=pd.read_csv(genespath);genes=genes[genes.status.eq('evaluated')]
        genes=genes.reindex(genes.equal_donor_delta_pred_mean.abs().sort_values(ascending=False).index)
        genes.to_csv(out/'genes_ranked.csv',index=False)
        pictures.append(save(rank_figure(genes,'gene','equal_donor_delta_pred_mean','Gene deletions · equal donor weights'),out,'gene_ranking','Ranked all-gene deletion effects'))
        donors=pd.read_csv(genespath.parent/'donor_level_prediction_deltas.csv.gz',dtype={'sample_id':str})
        names=genes.head(12).gene.tolist();donors=donors[donors.gene.isin(names)]
        for start in range(0,len(names),6):
            fig,values=donor_grid(donors,'delta_pred_mean',names[start:start+6],group+' · top gene deletions',columns=3,shared_y=True)
            if fig is not None:pictures.append(save(fig,out,f'top_genes_{start//6+1}','Top gene donor age changes'))
    bhlhe=root/'bhlhe40'
    if all((bhlhe/d/'complete.json').exists() for d in ['delete','overexpress']):pictures+=bhlhe40_plots(group,out)
    downloads=''.join(f'<li><a href="{p.name}">{p.name}</a></li>' for p in sorted(out.glob('*.csv')))
    page(out,group+' · age perturbations','<section><p>One fitted age model and verified baseline. Positive changes indicate older predictions; negative changes indicate younger predictions. Ranking uses absolute equal-donor mean changes. Token coverage refers to retained model inputs, not RNA detection percentage.</p></section><section><div class="plots">'+''.join(pictures)+'</div></section><section><h2>Data</h2><ul>'+downloads+'</ul></section>')
    return out/'index.html'


def bhlhe40_plots(group,out):
    pictures=[];frames=[]
    for direction in ['delete','overexpress']:
        root=RUN/'outputs'/group/'bhlhe40'/direction
        prepared=read(root/'prepared.json');m=read(root/f'bhlhe40_{direction}_manifest.json')
        donors=pd.read_csv(m['donor_level_prediction_deltas_csv']);donors['direction']=direction;frames.append(donors)
        fig,values=donor_grid(donors,'delta_pred_mean',prepared['programs'],'BHLHE40 proportional '+direction,columns=3,shared_y=True)
        values.to_csv(out/f'bhlhe40_{direction}_donor_values.csv',index=False)
        pictures.append(save(fig,out,'bhlhe40_'+direction+'_donor_dots','BHLHE40 proportional '+direction))
    allrows=pd.concat(frames,ignore_index=True);allrows.to_csv(out/'bhlhe40_all_donor_effects.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    baseline=allrows.groupby('sample_id').agg(true_mean=('true_mean','first'),base_pred_mean=('base_pred_mean','first')).sort_values('true_mean')
    spread=allrows.groupby('sample_id').base_pred_mean.agg(lambda x:x.max()-x.min())
    if spread.max()>1e-5:raise ValueError('BHLHE40 experiments do not share a baseline')
    ax.plot(baseline.true_mean,baseline.base_pred_mean,'o-',color='grey',label='Baseline',lw=2)
    for (direction,label),part in allrows.groupby(['direction','perturbation_label']):
        if '_signed_strength_' not in label:continue
        part=part.sort_values('true_mean');ax.plot(part.true_mean,part.perturbed_pred_mean,'.-',label=label,lw=1)
    ax.set(xlabel='Donor true mean log2(PCW)',ylabel='Donor mean predicted log2(PCW)',title='BHLHE40 proportional edits and shared baseline');ax.legend(fontsize=7,bbox_to_anchor=(1,1),loc='upper left')
    pictures.append(save(fig,out,'bhlhe40_baseline_comparison','BHLHE40 · donor mean-age progression'))
    return pictures


def classification_report():
    root=RUN/'outputs/delight_fm';out=root/'web_summary';out.mkdir(parents=True,exist_ok=True)
    annotations=pd.read_csv(RUN/'exported/delight_fm/obs.csv.gz',usecols=['cell_id','BroadCellType','sample_id','PCW']).set_index('cell_id')
    allscores=[];umaps=[];coverage=[]
    for kind in ['markers','regulons']:
        for manifest in (root/kind).glob('shard_*/v12_delete_manifest.json'):
            from screens import committed
            if not committed(manifest.parent,'delight_fm'):continue
            m=read(manifest);summary=pd.read_csv(m['summary_csv'])
            for row in summary.itertuples():
                label=row.perturbation_label;safe=re.sub(r'[^A-Za-z0-9_.-]+','_',label)
                path=manifest.parent/'umap_perturbation_plots'/f'v12_delete_{safe}_umap.csv.gz'
                if not path.exists():continue
                frame=pd.read_csv(path);metadata=annotations.reindex(frame.cell_id)
                if metadata.BroadCellType.isna().any() or frame.cell_id.duplicated().any():raise ValueError('Incomplete classifier annotations')
                scores=grouped_cell_effects(dict(input_gene=row.input_gene,perturbation_label=label,perturbation_kind=row.perturbation_kind),
                    metadata.BroadCellType.to_numpy(),frame.gene_present.to_numpy(),frame.delta_lineage_score.to_numpy(),frame.embedding_l2_shift.to_numpy())
                allscores.extend(dict(s,kind=kind) for s in scores)
                coverage.append(dict(perturbation=label,kind=kind,n_cells=len(frame),target_present=int(frame.gene_present.sum())))
                fig,ax=plt.subplots(figsize=(4.2,3.6),layout='constrained')
                image=ax.scatter(frame.umap_1,frame.umap_2,c=frame.embedding_l2_shift,s=1,cmap='viridis',rasterized=True,linewidths=0)
                ax.set(xlabel='UMAP 1',ylabel='UMAP 2',title=label);fig.colorbar(image,ax=ax,label='Embedding L2 shift')
                umaps.append(save(fig,out,kind+'_'+safe+'_UMAP_L2',label+' · embedding L2 change'))
    if not allscores:return None
    scores=pd.DataFrame(allscores);scores.to_csv(out/'model_cell_type_scores.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'perturbation_coverage.csv',index=False)
    pictures=[]
    for kind in ['markers','regulons']:
        for metric,suffix in [('lineage_contribution_score_mean','all_cells'),('present_lineage_contribution_score_mean','gene_present_cells')]:
            part=scores[scores.kind.eq(kind)].copy();part.loc[part.n_cells_with_gene.eq(0),metric]=np.nan
            if part.empty:continue
            matrix=part.pivot(index='perturbation_label',columns='lineage',values=metric)
            for transformed,stem in [(matrix,'signed_contribution'),((matrix-matrix.mean())/matrix.std(ddof=1).replace(0,np.nan),'z_score')]:
                transformed=transformed.dropna(how='all')
                if transformed.empty:continue
                rows,columns=correlation_linkages(transformed)
                if rows is not None:transformed=transformed.iloc[leaves_list(rows)]
                if columns is not None:transformed=transformed.iloc[:,leaves_list(columns)]
                pictures.append(heatmap(transformed,out,kind+'_'+stem+'_'+suffix,kind+' · '+stem.replace('_',' ')+' · '+suffix.replace('_',' ')))
    page(out,'DELIGHT-FM · markers and CollecTRI regulons','<section><p>One saved classifier and verified baseline. Positive contribution means deletion lowers confidence in the annotated model class; negative contribution means confidence increases. Regulons use CollecTRI only and jointly delete TF plus activating/repressing target membership. No SCENIC+ merged screen is run. Auxiliary age scores are not labelled log2(PCW).</p></section><section><div class="plots">'+''.join(pictures)+'</div></section><section><h2>UMAP embedding L2 changes</h2><details><summary>Show individual perturbation UMAPs</summary><div class="plots">'+''.join(umaps)+'</div></details></section>')
    return out/'index.html'


def main():
    pages=[training_report()]
    if (RUN/'outputs/delight_fm/regulons').exists():pages.append(classification_report())
    for group in ['immune','epithelium_colon','epithelium_ileum','stromal']:pages.append(age_report(group))
    out=RUN/'outputs/web_summary';out.mkdir(parents=True,exist_ok=True)
    import os
    links='<ul>'+''.join(f'<li><a href="{os.path.relpath(p,out)}">{html.escape(p.parent.parent.name if p.parent.name=="web_summary" else "Training and downstream sweeps")}</a></li>' for p in pages if p)+'</ul>'
    page(out,'v12 DELIGHT-FM · model and perturbation summaries','<section>'+links+'<p>Individual figures and value tables are stored beside each model summary. Reports remain separate to keep loading manageable.</p></section>')
    atomic_json(out/'report_provenance.json',dict(plotting_source_sha256={name:digest(Path(__file__).with_name(name))
        for name in ['plots.py','figure_style.py','donor_plotting.py','report_assets.py']},
        plan_sha256=digest(RUN/'diagnostics/plan.json')))
    print('Wrote',out/'index.html',flush=True)
