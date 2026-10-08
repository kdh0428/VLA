"""P1-C design: build log split + leakage stats (CPU only, reads existing outputs, writes design dir)."""
import json, glob, os, csv, hashlib, collections, yaml
D='/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1C_new_log_replication/design_20261008'
S=D+'/work'
N='/root/VLA/autovla/dataset/nuplan'
poc=sorted(yaml.safe_load(open('/root/VLA/autovla_misalignment_poc/configs/scene_filter_navtest_subset.yaml'))['log_names'])
nt=set(yaml.safe_load(open('/root/VLA/autovla/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml'))['log_names'])
alltest=sorted(f[:-4] for f in os.listdir(N+'/navsim_logs/test'))
frame=[r for r in json.load(open(S+'/frame_labels_5090.json')) if r['gpu']=='5090']
by=collections.defaultdict(list)
for r in frame: by[r['log']].append(r)
tok2shard={}
for f in glob.glob(N+'/_stream_gpu*/tokens_*.json'):
    sh=int(f.rsplit('_',1)[1][:-5])
    for t in json.load(open(f)): tok2shard[t]=sh
nh=json.load(open(S+'/navhard_logs.json')); nhs=set(nh['navhard_src'])
lu=json.load(open(S+'/log_usage_by_dir.json')); tu=json.load(open(S+'/token_usage_by_dir.json'))
use=collections.defaultdict(set)
for k,v in lu.items():
    for l in v: use[l].add(k)
for k,v in tu.items():
    for l in v: use[l].add(k)
# PoC mechanism usage (scenes in exp 6-11/32)
ed=collections.defaultdict(lambda: collections.Counter())
for l in open('/root/VLA/autovla_misalignment_poc/outputs/equal_distance_perturbation/records.jsonl'):
    r=json.loads(l); ed[r['log']][r['group']]+=1
fe=collections.Counter(json.loads(l)['log_name'] for l in open('/root/VLA/autovla_misalignment_poc/outputs/full_extract/records.jsonl'))
drive=lambda l: l.rsplit('_',2)[0]
dayveh=lambda l: l[:10]+'_'+l.split('_')[1]
pocd=set(map(drive,poc)); pocdv=set(map(dayveh,poc))
sb=set(os.listdir(N+'/sensor_blobs/test'))
rows=[]
def h(x): return hashlib.sha256(x.encode()).hexdigest()
for l in alltest:
    r=dict(log=l, drive=drive(l), day_vehicle=dayveh(l), in_navtest=l in nt)
    if l in poc: prior='poc_mechanism_dev'
    elif l in by: prior=by[l][0]['prior']+'_selection'
    elif l in nt: prior='?'
    else: prior='never_used_non_navtest'
    r['prior_use']=prior
    scen=by.get(l,[])
    r['shard']=(sorted({tok2shard.get(s['token']) for s in scen})[0] if scen else ('0-5' if l in sb or l in poc else 'unknown'))
    r['n_scenes_navtest']=len(scen) if scen else fe.get(l,0)
    r['n_frame_Aminus_5090']=sum(s['a_minus'] for s in scen) if scen else None
    r['n_frame_mismatch_Aplus_5090']=sum((s['t_star'] is not None) and not s['a_minus'] for s in scen) if scen else None
    r['n_frame_nomismatch_5090']=sum(s['t_star'] is None for s in scen) if scen else None
    r['poc_mech_Aminus_scenes']=ed[l]['A-'] if l in poc else None
    r['poc_mech_Aplus_scenes']=ed[l]['A+'] if l in poc else None
    r['navhard_source']=l in nhs
    r['drive_overlaps_poc']=(drive(l) in pocd) and l not in poc
    r['dayveh_overlaps_poc']=(dayveh(l) in pocdv) and l not in poc
    r['images_local']=l in sb
    r['n_output_dirs_using']=len(use.get(l,()))
    r['output_dirs_using']=';'.join(sorted(use.get(l,())))
    rows.append(r)
# roles
for r in rows:
    if r['prior_use']=='poc_mechanism_dev': r['role']='dev_discovery'
    elif r['prior_use']=='never_used_non_navtest': r['role']='reserve_non_navtest_not_used'
    elif r['drive_overlaps_poc']: r['role']='excluded_drive_adjacent_to_dev'
    else: r['role']='eval'
# pilot: 2 logs from drive-adjacent (excluded) set with >=1 frame A-, smallest scene count first, ties by sha256
cand=[r for r in rows if r['role']=='excluded_drive_adjacent_to_dev' and (r['n_frame_Aminus_5090'] or 0)>=1]
cand.sort(key=lambda r:(r['n_scenes_navtest'],h(r['log'])))
for r in cand[:2]: r['role']='pilot_harness_check'
# eval order (for sequential execution / stopping): sha256 of drive then log
for r in rows: r['eval_order_key']=h('P1C-20261008|'+r['drive']+'|'+r['log'])[:12] if r['role']=='eval' else ''
cols=['log','drive','day_vehicle','role','prior_use','in_navtest','shard','n_scenes_navtest','n_frame_Aminus_5090','n_frame_mismatch_Aplus_5090','n_frame_nomismatch_5090',
      'poc_mech_Aminus_scenes','poc_mech_Aplus_scenes','navhard_source','drive_overlaps_poc','dayveh_overlaps_poc','images_local','n_output_dirs_using','eval_order_key','output_dirs_using']
rows.sort(key=lambda r:(r['role'],r['log']))
with open(D+'/split.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); [w.writerow({c:r[c] for c in cols}) for r in rows]
# summaries
C=collections.Counter(r['role'] for r in rows); print(C)
ev=[r for r in rows if r['role']=='eval']
print('eval logs',len(ev),'drives',len({r['drive'] for r in ev}),'dayveh',len({r['day_vehicle'] for r in ev}),'scenes',sum(r['n_scenes_navtest'] for r in ev),
      'A-',sum(r['n_frame_Aminus_5090'] for r in ev),'A+mm',sum(r['n_frame_mismatch_Aplus_5090'] for r in ev),'nomm',sum(r['n_frame_nomismatch_5090'] for r in ev))
print('eval navhard src',sum(r['navhard_source'] for r in ev),'dayveh overlap',sum(r['dayveh_overlaps_poc'] for r in ev),'prior',collections.Counter(r['prior_use'] for r in ev))
dA=collections.Counter()
for r in ev: dA[r['drive']]+=r['n_frame_Aminus_5090']
print('drives with A-',sum(v>0 for v in dA.values()),'A- per drive dist',sorted(dA.values()))
print('shards for eval',sorted({r['shard'] for r in ev}))
ne=[r for r in ev if not r['navhard_source']]
print('eval non-navhard logs',len(ne),'drives',len({r['drive'] for r in ne}),'A-',sum(r['n_frame_Aminus_5090'] for r in ne),'scenes',sum(r['n_scenes_navtest'] for r in ne))
nd=[r for r in ev if not r['dayveh_overlaps_poc']]
print('eval dayveh-disjoint logs',len(nd),'drives',len({r['drive'] for r in nd}),'A-',sum(r['n_frame_Aminus_5090'] for r in nd))
print('pilot',[(r['log'],r['n_scenes_navtest'],r['n_frame_Aminus_5090'],r['shard']) for r in rows if r['role']=='pilot_harness_check'])
rv=[r for r in rows if r['role'].startswith('reserve')]
print('reserve', [(r['log'], r['drive_overlaps_poc'], r['dayveh_overlaps_poc'], r['images_local']) for r in rv])
# drive overlap of reserve with eval
evd={r['drive'] for r in ev}; evdv={r['day_vehicle'] for r in ev}
print('reserve drive in eval', sum(r['drive'] in evd for r in rv), 'dayveh in eval', sum(r['day_vehicle'] in evdv for r in rv))
# scenes per eval log
json.dump(dict(eval=[{k:r[k] for k in ('log','drive','day_vehicle','n_scenes_navtest','n_frame_Aminus_5090','n_frame_mismatch_Aplus_5090','n_frame_nomismatch_5090','navhard_source','dayveh_overlaps_poc','shard')} for r in ev]),open(S+'/eval_pool.json','w'))
