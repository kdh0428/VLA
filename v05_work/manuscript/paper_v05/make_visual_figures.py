#!/usr/bin/env python3
"""Paper-sized figures from the supplied post-hoc visual-evidence package.

Uses stored outputs only. No inference, interpolation, or new statistical test.
"""
from pathlib import Path
import csv, json, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'visual_inputs'
OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,
    'axes.titlesize':8,'axes.labelsize':7.3,'xtick.labelsize':7,
    'ytick.labelsize':7,'legend.fontsize':6.8,'pdf.fonttype':42,
    'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6})
BLUE='#2474B5'; ORANGE='#D56938'; GREEN='#23866C'; GRAY='#62666B'; PURPLE='#875AA1'

def rows(rel):
    with (DATA/rel).open() as f:return list(csv.DictReader(f))

def save(fig,name):
    for ext in ['pdf','png','svg']:
        buf=io.BytesIO()
        fig.savefig(buf,format=ext,dpi=300,facecolor='white')
        payload=buf.getvalue()
        if ext=='png':
            Image.open(io.BytesIO(payload)).load()
        elif ext=='pdf':
            from pypdf import PdfReader
            assert len(PdfReader(io.BytesIO(payload)).pages)==1
        target=OUT/(name+'.'+ext)
        temporary=target.with_suffix(target.suffix+'.tmp')
        temporary.write_bytes(payload)
        temporary.replace(target)
    plt.close(fig)

def clean(ax):
    ax.grid(axis='y',color='#e2e5e8',lw=.45,zorder=0)
    ax.set_axisbelow(True)

def cases():
    names=['improved_1','unchanged_severe_1','worsened_1']
    titles=['(a) Lower scene severe-event rate','(b) Same scene severe-event rate',
            '(c) Higher scene severe-event rate']
    fig=plt.figure(figsize=(6.875,3.05))
    gs=fig.add_gridspec(2,3,height_ratios=[.90,1.55],left=.07,right=.985,
                        bottom=.26,top=.91,hspace=.13,wspace=.33)
    for col,(name,title) in enumerate(zip(names,titles)):
        d=json.loads((DATA/'cases'/name/'case.json').read_text())
        image_path=next((DATA/'cases'/name).glob('*.jpg'))
        ax=fig.add_subplot(gs[0,col]);ax.imshow(Image.open(image_path));ax.axis('off')
        ax.set_title(title,pad=4,fontweight='bold',fontsize=7.0)
        ax=fig.add_subplot(gs[1,col]);clean(ax)
        gt=np.asarray(d['gt_xy']);a=int(d['anchor'])
        p=np.asarray(d['cells']['NAT_REF']['poses'])[:,:2]
        q=np.asarray(d['cells']['NAT_NSRC']['poses'])[:,:2]
        assert np.array_equal(p[:a],q[:a]),name+' emitted prefix differs'
        ax.plot(np.r_[0,gt[:,0]],np.r_[0,gt[:,1]],'--',c=GRAY,lw=1.1)
        ax.plot(p[a-1:,0],p[a-1:,1],c=BLUE,lw=1.5,marker='.',ms=3)
        ax.plot(q[a-1:,0],q[a-1:,1],c=ORANGE,lw=1.3,marker='.',ms=3)
        ax.plot(np.r_[0,p[:a,0]],np.r_[0,p[:a,1]],c='#25292d',lw=1.7)
        ax.scatter([p[a-1,0]],[p[a-1,1]],c='#25292d',s=12,zorder=5)
        ax.set_aspect('equal',adjustable='datalim')
        ax.set_xlabel('Forward position (m)',labelpad=2)
        ax.set_ylabel('Left (m)',labelpad=1)
        m=d['metrics']
        ax.text(.02,.98,f"FDE: {m['fde5_NAT_REF']:.2f} / {m['fde5_NAT_NSRC']:.2f} m",
            transform=ax.transAxes,va='top',fontsize=6.6,
            bbox={'facecolor':'white','alpha':.88,'edgecolor':'none','pad':1})
    handles=[Line2D([],[],c='#25292d',lw=1.6,label='Shared emitted prefix'),
             Line2D([],[],c=BLUE,lw=1.6,label='GT latest context'),
             Line2D([],[],c=ORANGE,lw=1.6,label='Normal-source latest context'),
             Line2D([],[],c=GRAY,ls='--',lw=1.2,label='GT trajectory')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.51,.005),
               ncol=2,frameon=False,columnspacing=1.8,handlelength=2.1)
    save(fig,'driving_cases')

def point(ax,r,y,color,marker='o',textoffset=6):
    est=float(r['estimate'])
    if r.get('ci_drawn')=='True' and r.get('ci_low') and r.get('ci_high'):
        ax.errorbar(est,y,xerr=[[est-float(r['ci_low'])],[float(r['ci_high'])-est]],
                    fmt=marker,color=color,ms=4.2,capsize=2,lw=1.1)
    else:
        ax.plot(est,y,marker=marker,ms=4.7,mfc='white',mec=color,mew=1.1)
    ax.annotate(f'{est:+.1f}',(est,y),xytext=(0,textoffset),textcoords='offset points',
                ha='center',va='bottom' if textoffset>0 else 'top',fontsize=6.8,color=color)

def effects():
    data=rows('EFFECTS_FOR_PLOTTING.csv');byid={r['claim_id']:r for r in data}
    fig,axes=plt.subplots(1,2,figsize=(6.875,1.95))
    fig.subplots_adjust(left=.125,right=.98,bottom=.24,top=.81,wspace=.40)
    specs=[('REP.P0A.DEV.RN.Am','REP.P0A.DEV.RV.Am'),
           ('REP.P0A.P1C.RN.F','REP.P0A.P1C.RV.F'),
           ('REP.P0A.P1C.RN.P','REP.P0A.P1C.RV.P')]
    ax=axes[0]
    for y,ids in zip([2,1,0],specs):
        for k,(rid,color) in enumerate(zip(ids,[BLUE,ORANGE])):
            point(ax,byid[rid],y+.13*(-1 if k==0 else 1),color)
    ax.set_yticks([2,1,0],['Dev F\n52 scenes','P1-C F\n105 scenes','Weighted E\n353 sampled'])
    ax.set_ylim(-.55,2.65);ax.set_xlim(-57,44);ax.axvline(0,c='#999999',ls=':',lw=.8)
    ax.set_xlabel('Severe-event difference (pp)')
    ax.set_title('(a) Repeated schedules',fontweight='bold',pad=16)
    ax.legend(handles=[Line2D([],[],color=BLUE,marker='o',label='RN: correction'),
                       Line2D([],[],color=ORANGE,marker='o',label='RV: reinsertion')],
              loc='lower left',bbox_to_anchor=(-.03,1.00),frameon=False,ncol=2,fontsize=6.3,
              columnspacing=1,handlelength=1.2)
    ax=axes[1]
    for y,stratum in [(1,'F'),(0,'P')]:
        for k,(contrast,color) in enumerate([('C_GT',ORANGE),('C_NAT',BLUE)]):
            r=next(x for x in data if x['contrast']==contrast and
                   ('B2 F' in x['cohort'] if stratum=='F' else 'E4' in x['cohort']))
            point(ax,r,y+.16*(-1 if k==0 else 1),color,textoffset=-6 if k==0 else 6)
    ax.set_yticks([1,0],['B2 F\n100 scenes','Weighted E4\n312 sampled'])
    ax.set_ylim(-.55,1.65);ax.set_xlim(-2,35);ax.axvline(0,c='#999999',ls=':',lw=.8)
    ax.set_xlabel('Severe-event difference (pp)')
    ax.set_title('(b) Matched-prefix contrasts',fontweight='bold',pad=16)
    ax.legend(handles=[Line2D([],[],color=ORANGE,marker='o',label=r'$C_{GT}$'),
                       Line2D([],[],color=BLUE,marker='o',label=r'$C_{NAT}$')],
              loc='lower left',bbox_to_anchor=(-.03,1.00),frameon=False,ncol=2,
              columnspacing=2,handlelength=1.2)
    save(fig,'conditional_effects')

def motion_time():
    data=[r for r in rows('TIME_SERIES_SUMMARY.csv') if r['setting']=='S0'
          and r['axis']=='step_after_anchor' and r['sample'].startswith('common')]
    data.sort(key=lambda r:float(r['step_after_anchor']))
    fig,axes=plt.subplots(1,2,figsize=(6.875,2.0))
    fig.subplots_adjust(left=.08,right=.985,bottom=.27,top=.77,wspace=.32)
    f=[r for r in data if r['stratum_sel']=='A-'];t=np.array([float(r['step_after_anchor'])/2 for r in f])
    ax=axes[0]
    for key,label,color,ls in [('D_model','Model',BLUE,'-'),('D_CM','Constant motion',ORANGE,'--'),
                              ('D_Geo','Coordinate accumulation',GREEN,':')]:
        ax.plot(t,[float(r[key]) for r in f],label=label,c=color,ls=ls,marker='o',ms=3,lw=1.4)
    ax.set_title('(a) Paired response: failure-selected F',fontweight='bold',fontsize=7.7)
    ax.set_ylabel(r'Response $D(h)$ (m)');ax.set_ylim(-.05,2.4)
    ax.legend(loc='upper left',frameon=False,fontsize=6.8)
    ax=axes[1]
    for key,label,color,ls in [('A-','F: 49 scenes',BLUE,'-'),('A+','Accepted: 147 scenes',GRAY,':'),
                            ('pooled (A- + A+)','Selected set: 196 scenes',ORANGE,'--')]:
        q=[r for r in data if r['stratum_sel']==key]
        ax.plot(t,[float(r['E_CM']) for r in q],label=label,c=color,ls=ls,marker='o',ms=3,lw=1.4)
    ax.axhline(0,c='#999999',lw=.7);ax.set_ylim(-.26,1.02)
    ax.set_title('(b) Excess over constant motion',fontweight='bold',fontsize=7.7)
    ax.set_ylabel(r'$D_{model}(h)-D_{CM}(h)$ (m)')
    ax.legend(loc='upper left',frameon=False,fontsize=6.8)
    for ax in axes:
        clean(ax);ax.set_xlabel('Time after perturbed token (s)');ax.set_xticks([0,.5,1,1.5,2,2.5])
    save(fig,'time_resolved_response')

def windows():
    data=[r for r in rows('tables/V2c_window_results.csv') if r['group']=='A-']
    fig,ax=plt.subplots(figsize=(3.25,1.65))
    fig.subplots_adjust(left=.17,right=.97,bottom=.31,top=.90)
    normal=next(r for r in data if r['condition']=='normal')
    full=next(r for r in data if r['condition']=='gt_history')
    points=sorted([r for r in data if r['condition'].startswith('win')],
                  key=lambda r:int(r['condition'].removeprefix('win')))
    xx=[0]+[int(r['condition'].removeprefix('win')) for r in points]
    yy=[float(normal['scene_mean'])*100]+[float(r['scene_mean'])*100 for r in points]
    ax.scatter(xx,yy,c=BLUE,s=20,zorder=3)
    ax.axhline(float(full['scene_mean'])*100,c=GRAY,ls='--',lw=1,
               label=f"Full-GT: {float(full['scene_mean'])*100:.1f}%")
    ax.annotate(f'{yy[0]:.1f}%',(0,yy[0]),xytext=(4,0),textcoords='offset points',fontsize=7)
    j=xx.index(3);ax.annotate(f'{yy[j]:.1f}%',(3,yy[j]),xytext=(0,7),
                             textcoords='offset points',ha='center',fontsize=7)
    ax.set_xticks(range(9));ax.set_ylim(0,52)
    ax.set_xlabel('Fixed GT window width (0 = Normal)',labelpad=3)
    ax.set_ylabel('Severe events (%)',labelpad=3)
    clean(ax);ax.legend(loc='upper right',frameon=False,fontsize=6.5)
    save(fig,'retained_window')

def spatial():
    data=rows('tables/V4_paired_units.csv')
    final=list(csv.DictReader((ROOT/'evidence/FINAL_PAPER_NUMBERS.csv').open()))
    fig,axes=plt.subplots(1,2,figsize=(6.875,2.05))
    fig.subplots_adjust(left=.08,right=.985,bottom=.26,top=.84,wspace=.44)
    ax=axes[0]
    cols=[f'dev_cm_t{j:02d}' for j in range(80)]
    for cfg,ls in [('r4e1','-'),('r1e1','--')]:
        for contrast,color in [('R-N',BLUE),('C-F',ORANGE)]:
            q=[r for r in data if r['config']==cfg and r['contrast']==contrast and r['eligible']=='True']
            assert len(q)==240
            vals=np.array([[float(r[k]) for k in cols] for r in q])
            ax.plot(range(80),vals.mean(0),c=color,ls=ls,lw=1.25,label=f'{contrast}, {cfg}')
    ax.axvspan(8,20,color='#f0f1f3',zorder=0)
    ax.set_title('(a) Actual paired TCP displacement',fontweight='bold',fontsize=7.8)
    ax.set_xlabel('Control step');ax.set_ylabel('Translation difference (cm)')
    ax.set_xlim(0,79);ax.set_xticks([0,20,40,60,79]);clean(ax)
    ax.legend(loc='upper left',frameon=False,fontsize=6.4,ncol=1)
    ax=axes[1]
    ax.axvspan(-10,10,color='#edf3f8',zorder=0)
    ax.axvline(-10,c='#a4b3c0',ls=':',lw=.8);ax.axvline(10,c='#a4b3c0',ls=':',lw=.8)
    ax.axvline(0,c='#999999',lw=.65)
    ys=[3,2,1,0]
    labels=[]
    for y,(cfg,con) in zip(ys,[('r1e4','R-N'),('r1e4','C-F'),('r4e1','R-N'),('r4e1','C-F')]):
        r=next(r for r in final if r['claim_id']==f'SVLA.P0B.P1D.{cfg}.{con}.pooled.physical')
        e=float(r['estimate']);lo=float(r['ci_low']);hi=float(r['ci_high']);c=BLUE if con=='R-N' else ORANGE
        ax.errorbar(e,y,xerr=[[e-lo],[hi-e]],fmt='o',c=c,ms=4,capsize=2,lw=1.1)
        from decimal import Decimal, ROUND_HALF_UP
        shown=Decimal(str(e)).quantize(Decimal('0.1'),rounding=ROUND_HALF_UP)
        ax.text(11.2,y,f'{shown:+}',va='center',fontsize=7,color=c)
        labels.append(f'{cfg}  {con}')
    ax.set_yticks(ys,labels);ax.set_ylim(-.6,3.6);ax.set_xlim(-13,16)
    ax.set_title('(b) Task-equal mean success effects',fontweight='bold',fontsize=7.8)
    ax.set_xlabel('Paired success difference (pp)')
    save(fig,'spatial_motion_success')

def heterogeneity():
    data=rows('tables/V3_B2_scene_effects.csv')
    drives=rows('tables/V3_B2_drive_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(6.875,2.0))
    fig.subplots_adjust(left=.08,right=.98,bottom=.25,top=.82,wspace=.24)
    import hashlib
    for ax,strat,color,title in zip(axes,['F','S'],[BLUE,ORANGE],['(a) Failure-selected F','(b) Sampled accepted stratum S']):
        q=[r for r in data if r['contrast']=='C_NAT' and r['stratum']==strat]
        dd=[r for r in drives if r['contrast']=='C_NAT' and r['stratum']==strat]
        w=np.array([float(r['weight']) for r in q])
        y=np.array([float(r['severe_scene_mean'])*100 for r in q])
        jitter=[(int(hashlib.sha256(r['scene'].encode()).hexdigest()[:8],16)/2**32-.5)*.8 for r in q]
        ax.scatter(jitter,y,s=6+12*w/np.max(w),c=color,alpha=.55,linewidths=0)
        for r in dd:
            ax.plot([.58,.72],[float(r['severe_wmean_pp'])]*2,c='#30363b',lw=.9)
        mean=np.average(y,weights=w)
        ax.axhline(mean,c=GRAY,lw=1,ls='--')
        ax.axhline(0,c='#bbbbbb',lw=.6)
        ax.text(.02,.96,f'{len(q)} scenes; {len(dd)} drives\nWeighted mean {mean:+.1f} pp',
                transform=ax.transAxes,va='top',fontsize=7,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':1})
        ax.set_title(title,fontweight='bold');ax.set_ylim(-30,105);ax.set_xlim(-.5,.82)
        ax.set_xticks([0,.65],['Scene effects','Drive summaries'])
        clean(ax)
    axes[0].set_ylabel(r'$C_{NAT}$: severe-event difference (pp)')
    axes[1].tick_params(labelleft=False)
    save(fig,'scene_drive_heterogeneity')

def tcp_cases():
    trace=rows('SPATIAL_SELECTED_TRACES.csv')
    selection=[r for r in rows('tables/V4_selected_cases.csv') if r['contrast']=='C-F' and r['rank']=='1']
    groups=['worse_success','same_success','better_success']
    fig,axes=plt.subplots(1,3,figsize=(6.875,2.1))
    fig.subplots_adjust(left=.08,right=.98,bottom=.27,top=.82,wspace=.37)
    for ax,group,title in zip(axes,groups,['(a) Success decreases','(b) Same success outcome','(c) Success increases']):
        sel=next(r for r in selection if r['group']==group)
        for arm,color in [('F',ORANGE),('C',BLUE)]:
            q=sorted([r for r in trace if r['case_group']==group and r['arm']==arm],key=lambda r:int(r['t']))
            assert len(q)==80
            x=np.array([float(r['tcp_x_m']) for r in q]);y=np.array([float(r['tcp_y_m']) for r in q])
            ax.plot(x,y,c=color,lw=1.3,label=f'{arm}')
            ax.scatter(x[0],y[0],c=color,s=10,marker='o');ax.scatter(x[-1],y[-1],c=color,s=22,marker='*')
        ax.set_aspect('equal',adjustable='datalim');clean(ax)
        ax.set_xlabel('TCP x (m)');ax.set_ylabel('TCP y (m)',labelpad=2)
        ax.set_title(title,fontweight='bold',fontsize=7.4)
        ax.text(.02,.96,f"F/C success: {sel['success_ref']}/{sel['success_treat']}\nseed {sel['seed']}",
                transform=ax.transAxes,va='top',fontsize=6.8,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.88,'pad':1})
    fig.legend(handles=[Line2D([],[],c=ORANGE,lw=1.5,label='F: perturbed execution and context'),
                        Line2D([],[],c=BLUE,lw=1.5,label='C: perturbed execution, corrected context')],
               loc='lower center',bbox_to_anchor=(.52,.005),ncol=2,frameon=False,fontsize=6.8,
               columnspacing=1.3,handlelength=1.7)
    save(fig,'spatial_tcp_cases')

def supplement_figures():
    # Show every available driving case at a size preserving the original labels.
    # These panels are generated independently from the same case JSON/images.
    names=['improved_1','improved_2','unchanged_severe_1','unchanged_severe_2','worsened_1']
    for name in names:
        d=json.loads((DATA/'cases'/name/'case.json').read_text());a=d['anchor']
        fig,axes=plt.subplots(1,3,figsize=(6.875,2.0),gridspec_kw={'width_ratios':[1.35,1,1.05]})
        fig.subplots_adjust(left=.02,right=.985,bottom=.27,top=.85,wspace=.35)
        axes[0].imshow(Image.open(next((DATA/'cases'/name).glob('*.jpg'))));axes[0].axis('off')
        axes[0].set_title('Stored front-camera frame',fontsize=7.5)
        gt=np.array(d['gt_xy']);p=np.array(d['cells']['NAT_REF']['poses'])[:,:2];q=np.array(d['cells']['NAT_NSRC']['poses'])[:,:2]
        for ax in axes[1:]:clean(ax)
        for pts,col,label in [(gt,GRAY,'GT trajectory'),(p,BLUE,'GT latest context'),(q,ORANGE,'Normal-source latest context')]:
            axes[1].plot(pts[:,1],pts[:,0],c=col,lw=1.2,ls='--' if col==GRAY else '-',label=label)
            axes[2].plot(np.arange(1,11)*.5,pts[:,0],c=col,lw=1.2,ls='--' if col==GRAY else '-')
        axes[1].plot(np.r_[0,p[:a,1]],np.r_[0,p[:a,0]],c='#222222',lw=1.7,label='Shared emitted prefix')
        axes[1].set_aspect('equal',adjustable='datalim');axes[1].set_xlabel('Left (m)');axes[1].set_ylabel('Forward (m)')
        axes[2].axvline(a*.5,c=GRAY,ls=':',lw=.8);axes[2].set_xlabel('Plan time (s)');axes[2].set_ylabel('Forward (m)')
        m=d['metrics'];axes[2].set_title(f"FDE: {m['fde5_NAT_REF']:.2f} / {m['fde5_NAT_NSRC']:.2f} m",fontsize=7.2)
        fig.legend(*axes[1].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.52,.015),ncol=4,frameon=False,fontsize=6.5,handlelength=1.5,columnspacing=1)
        save(fig,'case_detail_'+name)
    data=rows('TIME_SERIES_SUMMARY.csv')
    fig,axes=plt.subplots(1,2,figsize=(6.875,2.05))
    fig.subplots_adjust(left=.075,right=.98,bottom=.28,top=.80,wspace=.32)
    for ax,strat,title in zip(axes,['A-','pooled (A- + A+)'],['F: common 49 scenes','Selected set: common 196 scenes']):
        q=sorted([r for r in data if r['setting']=='S0' and r['axis']=='step_after_anchor' and r['sample'].startswith('common') and r['stratum_sel']==strat],key=lambda r:float(r['step_after_anchor']))
        t=[float(r['step_after_anchor'])/2 for r in q]
        for key,color,label in [('err_P_gt',BLUE,'Perturbed plan'),('err_U_gt',GRAY,'Unperturbed plan')]:
            ax.plot(t,[float(r[key]) for r in q],c=color,marker='o',ms=3,lw=1.2,label=label)
        ax.set_title(title,fontweight='bold');ax.set_xlabel('Time after perturbed token (s)');ax.set_ylabel('Distance to GT (m)');clean(ax)
    axes[0].legend(frameon=False,fontsize=7)
    save(fig,'gt_error_common')
    # Selection on success is explicitly descriptive, with no new test.
    pairs=rows('tables/V4_paired_units.csv');fig,ax=plt.subplots(figsize=(6.875,2.0))
    fig.subplots_adjust(left=.10,right=.98,bottom=.28,top=.85)
    for con,col,offset in [('R-N',BLUE,-.18),('C-F',ORANGE,.18)]:
        groups=[];ns=[]
        for val in [-1,0,1]:
            a=[float(r['area_dev_cmstep']) for r in pairs if r['config']=='r4e1' and r['contrast']==con and int(r['success_diff'])==val]
            groups.append(a);ns.append(len(a))
        bp=ax.boxplot(groups,positions=np.arange(3)+offset,widths=.28,patch_artist=True,showfliers=False,
                      medianprops={'color':col,'lw':1.4},boxprops={'facecolor':col,'alpha':.25,'edgecolor':col},
                      whiskerprops={'color':col},capprops={'color':col})
        for x,a,n in zip(np.arange(3)+offset,groups,ns):
            ax.text(x,np.percentile(a,75)+30,f'n={n}',ha='center',fontsize=6.8,color=col)
    ax.set_xticks([0,1,2],['Success decreases','Same success outcome','Success increases'])
    ax.set_ylabel('Paired TCP area (cm x step)');clean(ax)
    ax.legend(handles=[Line2D([],[],color=BLUE,lw=3,label='R-N'),Line2D([],[],color=ORANGE,lw=3,label='C-F')],frameon=False,ncol=2,loc='upper right')
    save(fig,'tcp_area_by_success')

if __name__=='__main__':
    cases();effects();motion_time();windows();spatial();heterogeneity();tcp_cases();supplement_figures()
    print('Visual figures generated from stored outputs; no inference or new tests.')
