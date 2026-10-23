#!/usr/bin/env python3
"""Rebuild the manuscript figure from the frozen final-number CSV (no model inference)."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
ROWS = {r['claim_id']: r for r in csv.DictReader((ROOT/'evidence/FINAL_PAPER_NUMBERS.csv').open())}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
                     'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
controls = [('Geo','Coordinate accumulation'),('CM','CM (with yaw)'),('CV0','Zero-turn velocity'),
            ('CTRV-K3','Turn rate + velocity'),('CTRA-2','Turn rate + acceleration'),
            ('E_prefix','Learned prefix')]
settings = [('S0','o','#155e86'),('SE','s','#bd6719'),('SF','^','#7751a3')]
fig, axes = plt.subplots(1,2,figsize=(9.5,3.55),sharey=True)
for ax,group,title in zip(axes,['A-','all'],['Failure-selected scenes','All selected scenes']):
    ax.axvline(0,color='#777777',lw=.9,zorder=1)
    for si,(setting,marker,color) in enumerate(settings):
        vals=[];lo=[];hi=[]
        for control,_ in controls:
            r=ROWS[f'GTFREE.{setting}.{group}.E_{control}']
            vals.append(float(r['estimate']));lo.append(float(r['ci_low']));hi.append(float(r['ci_high']))
        vals=np.array(vals);lo=np.array(lo);hi=np.array(hi)
        yy=np.arange(len(controls))+(si-1)*.19
        ax.errorbar(vals,yy,xerr=np.stack([vals-lo,hi-vals]),fmt=marker,color=color,
                    capsize=2,markersize=4.4,lw=1.1,zorder=3)
    ax.set_title(title,fontsize=11,fontweight='bold',pad=10)
    ax.set_xlabel('Endpoint-response excess (m)',labelpad=8)
    ax.grid(axis='x',color='#e4e6e8',lw=.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y',length=0,pad=8)
axes[0].set_yticks(np.arange(len(controls)),[x[1] for x in controls])
axes[0].invert_yaxis()
axes[0].set_xlim(-.25,5.6)
axes[1].set_xlim(-1.05,4.0)
legend=[Line2D([0],[0],marker=m,color=c,linestyle='none',label=l,markersize=5)
        for (s,m,c),l in zip(settings,['S0: original','SE: eager attention','SF: fp32 arithmetic'])]
fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.58,-.005),ncol=3,frameon=False,fontsize=9.5)
fig.subplots_adjust(left=.225,right=.985,top=.86,bottom=.25,wspace=.16)
(ROOT/'figures').mkdir(exist_ok=True)
fig.savefig(ROOT/'figures/motion_response_20261019.pdf',bbox_inches='tight',pad_inches=.08)
fig.savefig(ROOT/'figures/motion_response_20261019.png',dpi=180,bbox_inches='tight',pad_inches=.08)
print('Saved motion-response figure from 36 final-number rows.')
