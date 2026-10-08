import json, numpy as np, sys, collections
sys.path.insert(0,"/root/VLA/autovla_misalignment_poc/scripts/cross_domain_temporal")
from analyze_svla import boot, paired
from scipy.stats import binomtest, wilcoxon
O="/root/VLA/autovla_misalignment_poc/outputs/cross_domain_temporal_replication"
ch=np.zeros(3); n=0; oor=0; oor_units=0
lo,hi=None,None
from transformers import AutoProcessor
tt=AutoProcessor.from_pretrained("/root/VLA/spatialvla/spatialvla-4b-224-sft-fractal",trust_remote_code=True).action_tokenizer.translation_tokenizer
lo,hi=tt.token_start_idx,tt.token_end_idx
excl=collections.Counter()
for l in open(f"{O}/token_level/units.jsonl"):
    f=json.loads(l); R=f["R"]
    excl["kept"]+=len(f["units"])
    for u in f["units"]:
        n+=1; S=u["rows"]["normal"]
        for k in (1,2,3): ch[k-1]+= S[k][0]!=R[k][0]
        bad=any(not(lo<=row[k][0]<=hi) for row in u["rows"].values() for k in range(4))
        oor_units+=bad
    oor+=any(not(lo<=R[k][0]<=hi) for k in range(4))
print("step2/3/4 change", (ch/n).round(4), "n",n, "units with out-of-range trans id", oor_units, "frames R oor", oor)
E={}
for l in open(f"{O}/closed_loop/episodes.jsonl"):
    r=json.loads(l); E[(r["cond"],r["task"],r["seed"])]=r
keys=sorted({(t,s) for _,t,s in E})
a=[E[("feedback_opposite",*k)]["success"] for k in keys]; b=[E[("reverse_opposite",*k)]["success"] for k in keys]
g=sum(x and not y for x,y in zip(a,b)); l_=sum(y and not x for x,y in zip(a,b))
print("F-R opposite", np.mean(a)-np.mean(b), g,l_, binomtest(g,g+l_,.5).pvalue)
tr=lambda r: np.cumsum(np.array([x["action"][:3] for x in r["rec"]]),0)
def div(c,k): return float(np.linalg.norm(tr(E[(c,*k)])[-1]-tr(E[("natural",*k)])[-1]))
d=[div("reverse_opposite",k)-div("natural_gen",k) for k in keys]
print("div(rev_opp)-div(gen)", np.mean(d), boot([(k,x) for k,x in zip(keys,d)])["ci95"], wilcoxon(d).pvalue)
