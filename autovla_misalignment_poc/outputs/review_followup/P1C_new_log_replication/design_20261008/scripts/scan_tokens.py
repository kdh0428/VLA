import os, re, json, collections, sys, glob
O='/root/VLA/autovla_misalignment_poc/outputs'
t2l={}
for f in glob.glob(O+'/expanded_best_of_n/gpu*/records.jsonl')+glob.glob(O+'/heldout_best_of_n/gpu*/records.jsonl'):
    for l in open(f):
        r=json.loads(l); t2l[r['token']]=r['log']
for l in open(O+'/full_extract/records.jsonl'):
    r=json.loads(l); t2l[r['token']]=r['log_name']
print('tokens mapped',len(t2l), 'logs', len(set(t2l.values())))
json.dump(t2l,open('token2log.json','w'))
rx=re.compile(rb'(?<![0-9a-f])[0-9a-f]{16}(?![0-9a-f])')
use=collections.defaultdict(lambda: collections.defaultdict(set))
for top in sorted(os.listdir(O)):
    p=os.path.join(O,top)
    if not os.path.isdir(p) or top in ('full_extract',): continue
    for dp,dn,fn in os.walk(p):
        for f in fn:
            fp=os.path.join(dp,f)
            if os.path.getsize(fp)>800e6 or f.endswith(('.png','.pdf','.pt','.npy','.npz','.parquet')): continue
            with open(fp,'rb') as fh:
                for chunk in iter(lambda: fh.read(1<<24), b''):
                    for m in set(rx.findall(chunk)):
                        t=m.decode()
                        if t in t2l: use[top][t2l[t]].add(t)
out={k:{l:len(s) for l,s in v.items()} for k,v in use.items()}
json.dump(out,open('token_usage_by_dir.json','w'))
for k,v in sorted(out.items()): print(k,'logs',len(v),'scenes',sum(v.values()))
