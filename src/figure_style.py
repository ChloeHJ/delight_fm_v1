"""Shared visual conventions for DELIGHT-FM publication figures."""
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

POSITIVE='#b65050'
NEGATIVE='#277d8c'
HEATMAP='coolwarm'


def apply():
    plt.rcParams.update({'font.family':'sans-serif','font.size':9,
        'axes.spines.top':False,'axes.spines.right':False,
        'axes.titlesize':11,'axes.labelsize':9,'xtick.labelsize':9,'ytick.labelsize':9,
        'legend.fontsize':8,'figure.facecolor':'white','axes.facecolor':'white',
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':140})


def rank_figure(frame,label,value,title,limit=20):
    """Compact signed-effect bars for ranked perturbation reports."""
    plotted=frame.reindex(frame[value].abs().sort_values(ascending=False).head(limit).index)
    fig,ax=plt.subplots(figsize=(5,max(5,len(plotted)*.3)),layout='constrained')
    ax.barh(plotted[label],plotted[value],color=np.where(plotted[value]>0,POSITIVE,NEGATIVE))
    ax.invert_yaxis();ax.axvline(0,color='#637188',lw=.8)
    ax.set_title('\n'.join(textwrap.fill(line,48) for line in title.splitlines()),fontsize=13)
    ax.set_xlabel(textwrap.fill('Mean donor predicted age change (log2 PCW)',38),fontsize=12)
    ax.tick_params(axis='both',labelsize=11)
    return fig
