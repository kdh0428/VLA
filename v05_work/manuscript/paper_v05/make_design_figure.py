#!/usr/bin/env python3
"""Draw the exact primary matched-prefix intervention; no generated data."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
fig=plt.figure(figsize=(6.875,2.65))
ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
colors={'fixed':'#e9edf1','gt':'#dbeafa','nat':'#fbe5cd','own':'#dff0e8'}
def box(x,y,w,h,text,kind='fixed',fs=8.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.004,rounding_size=0.01',
                 facecolor=colors[kind],edgecolor='#6c7a86',linewidth=.65))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs)
ax.text(.015,.955,'Shared emitted prefix in every cell',fontweight='bold',fontsize=10)
xs=[.02,.25,.40,.56,.72];ws=[.20,.12,.13,.13,.255]
for x,w,text,c in zip(xs,ws,['Before t*','Forced f','Natural old','Natural latest','Own suffix from a'],
                       ['fixed','fixed','nat','nat','own']):box(x,.77,w,.12,text,c,8.2)
for i in range(4):ax.annotate('',xy=(xs[i+1]-.007,.83),xytext=(xs[i]+ws[i]+.005,.83),arrowprops={'arrowstyle':'->','lw':.7})
ax.text(.015,.66,'Context supplied to generate each suffix',fontweight='bold',fontsize=10)
ax.text(.31,.60,'Older position t*+1',ha='center',fontsize=8.5)
ax.text(.55,.60,'Latest position a-1',ha='center',fontsize=8.5)
ax.text(.84,.60,'Paired latest-token contrasts',ha='center',fontsize=8.5)
rows=[('GT_REF','GT','GT','gt','gt'),('GT_NSRC','GT','Normal','gt','nat'),
      ('NAT_REF','Natural','GT','nat','gt'),('NAT_NSRC','Natural','Normal','nat','nat')]
ys=[.45,.33,.17,.05]
for (lab,old,last,co,cl),y in zip(rows,ys):
    ax.text(.02,y+.043,lab,ha='left',va='center',fontsize=8.3)
    box(.225,y,.17,.085,old,co);box(.45,y,.20,.085,last,cl)
    ax.text(.417,y+.043,'+',ha='center',va='center',fontsize=9)
for y,lab in [(.43,'C_GT = GT_NSRC - GT_REF'),(.15,'C_NAT = NAT_NSRC - NAT_REF')]:
    ax.text(.68,y,lab,ha='left',va='center',fontsize=7.9)
ax.text(.68,.29,'Edited positions stay in later contexts.\nTheir values do not replace emitted actions.',
        ha='left',va='center',fontsize=7.4,color='#364652',linespacing=1.5)
(ROOT/'figures').mkdir(exist_ok=True)
fig.savefig(ROOT/'figures/matched_prefix_design.pdf',bbox_inches='tight',pad_inches=.03)
fig.savefig(ROOT/'figures/matched_prefix_design.png',dpi=200,bbox_inches='tight',pad_inches=.03)
print('Saved exact primary matched-prefix design.')
