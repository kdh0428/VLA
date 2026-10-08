import os, re, json, collections, sys
root='/root/VLA/autovla_misalignment_poc/outputs'
rx=re.compile(rb'20[0-9]{2}\.[0-9]{2}\.[0-9]{2}\.[0-9]{2}\.[0-9]{2}\.[0-9]{2}_veh-[0-9]+_[0-9]{5}_[0-9]{5}')
use=collections.defaultdict(set)
for top in sorted(os.listdir(root)):
    p=os.path.join(root,top)
    if not os.path.isdir(p): continue
    for dp,dn,fn in os.walk(p):
        if 'tensors' in dp.split(os.sep): continue
        for f in fn:
            fp=os.path.join(dp,f)
            if os.path.getsize(fp)>600e6 or f.endswith(('.png','.pdf','.pt','.npy','.npz','.parquet')): continue
            try:
                with open(fp,'rb') as fh:
                    for chunk in iter(lambda: fh.read(1<<24), b''):
                        for m in set(rx.findall(chunk)): use[top].add(m.decode())
            except Exception as e: print('err',fp,e,file=sys.stderr)
# full_extract records
for dp in ['/root/VLA/navhard/navhard_two_stage/synthetic_scenes_attributes.csv']:
    pass
json.dump({k:sorted(v) for k,v in use.items()}, open('log_usage_by_dir.json','w'))
for k,v in sorted(use.items()): print(k,len(v))
