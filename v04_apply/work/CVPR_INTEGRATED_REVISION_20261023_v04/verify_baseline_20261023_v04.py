"""CPU verification from the raw archives included in this package.
Requires numpy; run from any directory. Does not run inference or physics.
"""
from pathlib import Path
import csv, io, json, zipfile, re
import numpy as np

ROOT=Path(__file__).resolve().parent
checks={}
def check(name,ok):
    checks[name]=bool(ok)
    if not ok: raise AssertionError(name)

with zipfile.ZipFile(ROOT/'raw_packages/CVPR_VLA0_AND_CONTRIBUTION_20261020.zip') as z:
    key=next(n for n in z.namelist() if n.endswith('/run/pilot/records.jsonl'))
    records=[json.loads(l) for l in z.read(key).decode().splitlines()]
records=[r for r in records if r['status']=='OK']
check('pilot_149_observations',len(records)==149)
check('pilot_30_tasks',len({(r['suite'],r['task_index']) for r in records})==30)
values=[]
for r in records:
    s={}
    for a in ['R','L','O','OL']:
        ids=r['arms'][a]['ids']
        txt=''.join(' ' if t==220 else str(t-15) for t in ids if t not in (151643,151645))
        v=[int(x) for x in txt.split()]
        check(r['obs_id']+'/'+a+'/35_valid_numbers',len(v)==35 and all(0<=v<=1000 for v in v))
        s[a]=np.asarray(v,float).reshape(5,7)[:,:6]/1000
    dist=lambda a,b:float(np.abs(s[a]-s[b]).mean())
    A={a:dist(a,'R') for a in s}
    orig=np.asarray(r['reference']['audit']['numbers'],float).reshape(8,7)
    dose={}
    for a,j in [('L',3),('O',2)]:
        donor=np.asarray(r['donors'][str(j)]['vector'][:6],float)
        dose[a]=float(np.abs(donor-orig[j-1,:6]).mean()/1000)
    ratio=dose['L']/dose['O'] if dose['O'] else np.nan
    values.append({'primary':A['L']-A['O'],'direct_OL_L':dist('OL','L'),
                   'abs_scalar_response_difference':abs(A['OL']-A['L']),
                   'distance_matched':bool(.8<=ratio<=1.25)})
means={k:float(np.mean([r[k] for r in values])) for k in ['primary','direct_OL_L','abs_scalar_response_difference']}
check('primary_L_minus_O',abs(means['primary']-.01367874720357942)<1e-12)
check('direct_OL_L',abs(means['direct_OL_L']-.01522483221476510)<1e-12)
check('absolute_scalar_response_difference_is_different',abs(means['abs_scalar_response_difference']-.00709149888143177)<1e-12)
check('matched_subset_66',sum(r['distance_matched'] for r in values)==66)
body=(ROOT/'paper/paper_body.tex').read_text()
check('Table3_uses_direct_distance',r'$D(OL,L)^\dagger$' in body)
check('obsolete_scalar_notation_removed','A_{OL}-A_L' not in body)
check('distance_ratio_is_printed',r'0.8\leq d_L/d_O\leq1.25' in body)
check('direct_distance_definition',r'A_k=D(k,R)' in body)

rows=list(csv.DictReader((ROOT/'tables/vla0_closed_loop_8rows.csv').open()))
with zipfile.ZipFile(ROOT/'raw_packages/CVPR_VLA0_CLOSED_LOOP_FEASIBILITY_20261021_v02.zip') as z:
    prefix=next(n for n in z.namelist() if n.endswith('/run/attempt1/STATUS.json')).split('/run/')[0]
    def trace(u,a):
        d=np.load(io.BytesIO(z.read(f'{prefix}/run/attempt1/{u}/{a}/trace.npz')),allow_pickle=False)
        return {k:d[k] for k in d.files}
    for u in ['U1','U2','U3','U4']:
        n=trace(u,'N')
        for a in ['R','L','O']:
            t=trace(u,a); row=next(r for r in rows if r['unit']==u and r['edit']==a)
            ee=np.linalg.norm(t['eef_pos'][11:]-n['eef_pos'][11:],axis=1)*1000
            cmd=np.abs(t['action'][10:,:6]-n['action'][10:,:6]).mean(axis=1)
            if a=='R':
                check(u+'/noop_states_actions',t['state'].tobytes()==n['state'].tobytes() and t['action'].tobytes()==n['action'].tobytes())
            for k,v in [('mean_ee_mm_all48',ee.mean()),('mean_ee_mm_steps8to47',ee[8:].mean()),
                        ('final_ee_mm',ee[-1]),('mean_cmd_all48',cmd.mean())]:
                check(u+'/'+a+'/'+k,abs(float(row[k])-v)<1e-12)

supp=(ROOT/'paper/supplement_body.tex').read_text()
check('complete_supplement_A_to_O',len(re.findall(r'\\section\{',supp))==15)
for label in ['app:metric','app:stats','app:matched','app:motion','app:window','app:spatial','app:othermodels','app:provenance','app:visual']:
    check('preserved_original_label/'+label,'\\label{'+label+'}' in supp)
result={'all_passed':all(checks.values()),'checks':len(checks),'means_recomputed_from_raw_tokens':means,
        'scope':'CPU token and pose recomputation only; no fresh inference or physics.'}
(ROOT/'qa/BASELINE_RECOMPUTATION_20261023_v04.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
