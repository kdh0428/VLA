import os, sys, json, glob
sys.path.insert(0,'/root/VLA/autovla_misalignment_poc/scripts')
from analyze_action_history import unit_metrics
O='/root/VLA/autovla_misalignment_poc/outputs'
src=[('dev','expanded_best_of_n/gpu1','5090'),('dev','expanded_best_of_n_5090/gpu1','5090'),('dev','expanded_best_of_n/gpu0','3080Ti'),
     ('heldout','heldout_best_of_n/gpu1','5090'),('heldout','heldout_best_of_n_5090/gpu1','5090'),('heldout','heldout_best_of_n/gpu0','3080Ti')]
out={}
for prior,d,gpu in src:
    for l in open(f'{O}/{d}/records.jsonl'):
        r=json.loads(l)
        c=r['candidates'][0]
        m=unit_metrics(c['action_idx'],c['trajectory_pred'],c['entropy_steps'],r['gt'],r['trajectory_gt'],0)
        ts=next((k for k in range(10) if c['action_idx'][k]!=r['gt'][k]),None)
        row=dict(token=r['token'],log=r['log'],prior=prior,gpu=gpu,src=d,stub_cot=bool(r.get('stub_cot')),t_star=ts,a_minus=int(not m['recovery']),amp=int(m['amplification']))
        key=r['token']
        if gpu=='5090': out.setdefault(key+'|5090',row)
        else: out[key+'|3080']=row
json.dump(list(out.values()),open('frame_labels_all.json','w'))
# agreement 5090 vs 3080 where both
a=[(out[k], out[k.replace('|3080','|5090')]) for k in out if k.endswith('|3080') and k.replace('|3080','|5090') in out]
print('pairs',len(a),'Aminus agree',sum(x['a_minus']==y['a_minus'] for x,y in a)/max(1,len(a)),'t* agree',sum(x['t_star']==y['t_star'] for x,y in a)/max(1,len(a)))
print('A- 3080',sum(x['a_minus'] for x,y in a),'A- 5090',sum(y['a_minus'] for x,y in a),'both',sum(x['a_minus'] and y['a_minus'] for x,y in a))
n5=[v for k,v in out.items() if k.endswith('|5090')]
print('5090 scenes',len(n5),'A-',sum(v['a_minus'] for v in n5))
