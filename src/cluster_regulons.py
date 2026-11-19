"""Balanced magnitude-aware Ward clustering of signed donor age responses."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage,cut_tree,leaves_list
from scipy.spatial.distance import pdist,squareform
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import RUN
from common import config,atomic_json,digest
from figure_style import apply
from donor_plotting import donor_grid
from report_assets import embed_images
apply()


def features(values):
    values=np.asarray(values,dtype=float)
    if values.ndim!=2 or not np.isfinite(values).all():raise ValueError('Clustering requires a complete finite regulon × donor matrix')
    scale=float(np.median(np.sqrt(np.mean(values**2,axis=1))))
    if scale<=0:raise ValueError('Median regulon RMS must be positive')
    return np.arcsinh(values/scale),scale


def cluster(values,k):
    if not 2<=k<len(values):raise ValueError('Cluster count must be between 2 and n_regulons−1')
    transformed,scale=features(values)
    tree=linkage(transformed,method='ward',optimal_ordering=True)
    raw=cut_tree(tree,n_clusters=k).ravel();order=leaves_list(tree)
    mapping={c:j+1 for j,c in enumerate(dict.fromkeys(raw[order]))}
    labels=np.array([mapping[c] for c in raw])
    return transformed,scale,tree,labels,order


def run(group,donor_csv=None,out=None,k=None):
    if group=='delight_fm':raise ValueError('Age-pattern clustering is defined for lineage age models')
    source=Path(donor_csv) if donor_csv else RUN/'outputs'/group/'regulons/v12_delete_donor_level_prediction_deltas.csv'
    k=k or config()['clustering']['clusters']
    out=Path(out) if out else RUN/'outputs'/group/f'regulon_clusters_k{k}'
    out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    d=pd.read_csv(source,dtype={'sample_id':str})
    if 'perturbation_kind' in d:d=d[d.perturbation_kind.eq('regulon')]
    if d.duplicated(['perturbation_label','sample_id']).any():raise ValueError('Expected one response per regulon and biological donor')
    meta=d[['sample_id','true_mean','AgeGroup','n_cells']].drop_duplicates().sort_values(['true_mean','sample_id'])
    if meta.sample_id.duplicated().any():raise ValueError('Inconsistent donor metadata')
    matrix=d.pivot(index='perturbation_label',columns='sample_id',values='delta_pred_mean').reindex(columns=meta.sample_id)
    if len(meta)<3:raise ValueError('At least three donors are required for sensitivity analysis')
    x=matrix.to_numpy();f,scale,tree,labels,order=cluster(x,k)
    distances=squareform(pdist(f));assignments=pd.DataFrame(dict(regulon=matrix.index,cluster=labels,
        mean_delta=x.mean(axis=1),RMS_delta=np.sqrt(np.mean(x*x,axis=1))))
    assignments['TF']=assignments.regulon.str.split('_').str[0]
    reps=[]
    for c in range(1,k+1):
        ids=np.flatnonzero(labels==c)
        central=distances[np.ix_(ids,ids)].mean(axis=1)
        ranked=sorted(zip(ids,central),key=lambda pair:(pair[1],matrix.index[pair[0]]))
        chosen=[ranked[0][0]]
        second=next((i for i,_ in ranked[1:] if assignments.iloc[i].TF!=assignments.iloc[chosen[0]].TF),None)
        if second is not None:chosen.append(second)
        for rank,i in enumerate(chosen,1):reps.append(dict(cluster=c,regulon=matrix.index[i],representative_rank=rank))
    representatives=pd.DataFrame(reps)
    stability=[]
    for j,name in enumerate(matrix.columns):
        _,_,_,other,_=cluster(np.delete(x,j,axis=1),k)
        stability.append(dict(omitted_donor=name,adjusted_rand_index=adjusted_rand_score(labels,other)))
    scan=[]
    for count in range(2,min(12,len(x)-1)+1):
        lab=cut_tree(tree,n_clusters=count).ravel()
        scan.append(dict(k=count,silhouette=silhouette_score(f,lab),smallest_cluster=int(np.bincount(lab).min())))
    matrix.to_csv(out/'raw_prediction_change_matrix.csv')
    pd.DataFrame(f,index=matrix.index,columns=matrix.columns).to_csv(out/'clustering_features.csv')
    pd.DataFrame(tree,columns=['left','right','Ward_distance','n_leaves']).to_csv(out/'linkage.csv',index=False)
    assignments.to_csv(out/'cluster_assignments.csv',index=False)
    representatives.to_csv(out/'representatives.csv',index=False)
    pd.DataFrame(stability).to_csv(out/'leave_one_donor_out_stability.csv',index=False)
    pd.DataFrame(scan).to_csv(out/'resolution_scan.csv',index=False)
    meta.to_csv(out/'donor_metadata.csv',index=False)
    pd.DataFrame(dict(regulon=matrix.index[order],cluster=labels[order])).to_csv(out/'heatmap_row_order.csv',index=False)
    marked=set(representatives.regulon)|set(assignments.loc[assignments.TF.isin(config()['clustering']['focus_tfs']),'regulon'])
    pictures=[]
    def save(fig,stem):
        for ext in ['pdf','png']:fig.savefig(out/'figures'/f'{stem}.{ext}',dpi=160,bbox_inches='tight')
        plt.close(fig)
        pictures.append(f'<figure><img src="figures/{stem}.png"><figcaption><a href="figures/{stem}.pdf">PDF</a></figcaption></figure>')
    for values,stem,title in [(f,'feature_heatmap','Clustering features · asinh(Δ / common scale)'),(x,'raw_heatmap','Original signed prediction changes')]:
        fig,ax=plt.subplots(figsize=(8.25,7.5),layout='constrained')
        bound=max(float(np.max(np.abs(values))),1e-9)
        image=ax.imshow(values[order],aspect='auto',cmap='coolwarm',vmin=-bound,vmax=bound,interpolation='nearest')
        ax.set_xticks(range(len(meta)),[f'{r.sample_id}\n{r.true_mean:.2f}' for r in meta.itertuples()],rotation=65,ha='right',fontsize=8)
        ticks=[j for j,i in enumerate(order) if matrix.index[i] in marked]
        # Small adjacent clusters can place several representatives on consecutive
        # rows. Spread labels in display space and connect to their actual rows.
        ax.set_yticks([])
        separation=max(3.,len(order)/80.)
        positions=[]
        for tick in ticks:positions.append(max(float(tick),positions[-1]+separation if positions else 0.))
        for j in range(len(positions)-1,-1,-1):
            positions[j]=min(positions[j],len(order)-1 if j==len(positions)-1 else positions[j+1]-separation)
        for tick,position in zip(ticks,positions):
            ax.annotate(matrix.index[order[tick]],xy=(0,tick),xytext=(-.025,position),
                xycoords=('axes fraction','data'),textcoords=('axes fraction','data'),
                ha='right',va='center',fontsize=6,annotation_clip=False,
                arrowprops=dict(arrowstyle='-',color='grey',lw=.35))
        for c in range(1,k+1):
            positions=np.flatnonzero(labels[order]==c);ax.axhline(positions[-1]+.5,color='black',lw=.6)
        ax.set(title=title,xlabel='Donor · true mean log2(PCW)');fig.colorbar(image,ax=ax)
        save(fig,stem)
    pca=PCA(n_components=2);points=pca.fit_transform(f)
    fig,ax=plt.subplots(figsize=(8.25,5.25),layout='constrained')
    for c in range(1,k+1):
        choose=labels==c;ax.scatter(points[choose,0],points[choose,1],s=18,label=f'C{c} (n={choose.sum()})')
    texts=[]
    for i,label in enumerate(matrix.index):
        if label in marked:texts.append(ax.text(*points[i],label,fontsize=6))
    try:
        from adjustText import adjust_text
        adjust_text(texts,ax=ax,time_lim=3,arrowprops=dict(arrowstyle='-',lw=.4))
    except ImportError:pass
    ax.set(xlabel='PC1',ylabel='PC2',title='Balanced magnitude-aware Ward · PCA display only');ax.legend(fontsize=7)
    save(fig,'PCA')
    fig,axes=plt.subplots(int(np.ceil(k/2)),2,figsize=(8.25,3*int(np.ceil(k/2))),squeeze=False,layout='constrained')
    for ax,c in zip(axes.flat,range(1,k+1)):
        ids=np.flatnonzero(labels==c)
        for i in ids:ax.plot(meta.true_mean,x[i],color='grey',alpha=.15,lw=.6)
        ax.plot(meta.true_mean,np.median(x[ids],axis=0),color='black',lw=1.5,label='Cluster median')
        for name in representatives.loc[representatives.cluster.eq(c),'regulon']:
            ax.plot(meta.true_mean,x[matrix.index.get_loc(name)],'.-',label=name,lw=1)
        ax.axhline(0,color='grey',lw=.7);ax.set(title=f'C{c} · n={len(ids)}',xlabel='Donor true mean log2(PCW)',ylabel='Mean age change · log2(PCW)')
        ax.legend(fontsize=6)
    for ax in axes.flat[k:]:ax.set_visible(False)
    save(fig,'raw_profiles')
    fig,ax=plt.subplots(figsize=(4.5,2.7),layout='constrained')
    scanframe=pd.DataFrame(scan);ax.plot(scanframe.k,scanframe.silhouette,'o-');ax.axvline(k,ls='--',color='#b65050')
    ax.set(xlabel='Number of clusters',ylabel='Euclidean silhouette',title='Cluster resolution')
    save(fig,'resolution_scan')
    for c in range(1,k+1):
        names=list(dict.fromkeys(representatives.loc[representatives.cluster.eq(c),'regulon'].tolist()+
            assignments.loc[assignments.cluster.eq(c)&assignments.TF.isin(config()['clustering']['focus_tfs']),'regulon'].tolist()))
        fig,values=donor_grid(d,'delta_pred_mean',names,f'C{c} · representatives and named regulons',columns=3,shared_y=False)
        values.to_csv(out/f'C{c}_donor_values.csv',index=False);save(fig,f'C{c}_donor_dotplots')
    record=dict(group=group,k=k,regulons=len(matrix),donors=len(meta),common_scale=scale,
        common_scale_definition='Median across regulons of sqrt(mean squared donor age changes)',
        transform='asinh(delta / common_scale)',row_centering=False,per_row_normalization=False,
        linkage='Ward',distance='Euclidean',PCA_used_for_clustering=False,source=str(source),source_sha256=digest(source))
    atomic_json(out/'method.json',record)
    body=f'<h1>{group} · balanced magnitude-aware Ward · {k} clusters</h1><p>Common scale: {scale:.10g} log2(PCW). '
    body+='Signed effects are retained without row centring or variance normalization. Representatives minimize mean within-cluster feature distance. PCA is a display only.</p>'
    body+=assignments.groupby('cluster').size().rename('regulons').reset_index().to_html(index=False)
    body+=''.join(pictures)
    (out/'index.html').write_text(embed_images('<html><head><meta charset="utf-8"><style>body{font:15px Arial;max-width:1200px;margin:auto;padding:24px}img{max-width:100%}</style></head><body>'+body+'</body></html>',out))
    print(json.dumps(record,indent=2),flush=True)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--group',default='epithelium_ileum')
    parser.add_argument('--donor-csv',type=Path);parser.add_argument('--output-dir',type=Path);parser.add_argument('--clusters',type=int,default=12)
    args=parser.parse_args();run(args.group,args.donor_csv,args.output_dir,args.clusters)
