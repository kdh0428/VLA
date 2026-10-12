# Exploratory reversal-flag sensitivity (protocol §5 exclusion rule; Lead decision 4).
# Uses only module-level functions of the committed analyze_p1d.py (compact, pair, strat_mean, signflip, wil);
# the bootstrap below copies the vectorised branch of analyze_p1d.main.estimate verbatim (B_REPS, default_rng(0)).
import sys, json
import numpy as np
from collections import Counter
sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/review_followup/p1d")
import analyze_p1d as A
EP = "/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1D_dose_control/run_20261008/rollouts/main/episodes.jsonl"
C = {}
with open(EP) as f:
    for line in f:
        ep = json.loads(line); c = A.compact(ep)
        C[(ep["cond"], ep["task"], ep["seed"])] = c
tasks = sorted({k[1] for k in C}); keys = [(t, s) for t in tasks for s in sorted({k[2] for k in C if k[1] == t and k[0].endswith(":natural")})]
def nflip(cfg, mode, t, s):
    c = C.get((f"{cfg}:{mode}", t, s)); return None if c is None else sum(bool(i["flip"]) for i in c["inj"])
# flip-count distribution per arm
dist = {}
for cfg in ["r1e1", "r2e1", "r4e1", "r1e4"]:
    for mode in ["feedback", "corrected", "reverse", "feedbackD", "correctedD"]:
        v = [nflip(cfg, mode, t, s) for t, s in keys]; v = [x for x in v if x is not None]
        if v: dist[f"{cfg}:{mode}"] = {"n": len(v), "zero_flip": sum(x == 0 for x in v), "hist": dict(sorted(Counter(v).items()))}
print(json.dumps(dist, indent=0))
def boot(v, ks):
    tk = np.array([k[0] for k in ks]); ms = []
    for t in sorted(set(tk)):
        ix = np.nonzero(tk == t)[0]
        ms.append(v[ix][np.random.default_rng(0).integers(0, len(ix), size=(A.B_REPS, len(ix)))].mean(1))
    return np.mean(ms, 0)
def cv(cfg, x, y, metric, t, s):
    a, b = C.get((f"{cfg}:{x}", t, s)), C.get((f"{cfg}:{y}", t, s))
    return None if a is None or b is None else A.pair(a, b)[metric]
SPEC = [  # name, metric, (cfg_late, x, y), (cfg_early, x, y), perturbed arms that must have 0 flips
 ("H1", "area_dev_cmstep", ("r4e1", "reverse", "natural"), ("r1e1", "reverse", "natural"), [("r4e1", "reverse"), ("r1e1", "reverse")]),
 ("H2", "area_dev_cmstep", ("r1e4", "feedbackD", "correctedD"), ("r1e1", "feedback", "corrected"), [("r1e4", "feedbackD"), ("r1e4", "correctedD"), ("r1e1", "feedback"), ("r1e1", "corrected")]),
 ("S1", "success", ("r4e1", "reverse", "natural"), ("r1e1", "reverse", "natural"), [("r4e1", "reverse"), ("r1e1", "reverse")]),
 ("S2", "success", ("r1e4", "feedbackD", "correctedD"), ("r1e1", "feedback", "corrected"), [("r1e4", "feedbackD"), ("r1e4", "correctedD"), ("r1e1", "feedback"), ("r1e1", "corrected")]),
 ("S3", "success", ("r4e1", "feedback", "corrected"), ("r1e1", "feedback", "corrected"), [("r4e1", "feedback"), ("r4e1", "corrected"), ("r1e1", "feedback"), ("r1e1", "corrected")]),
 ("S4", "success", ("r1e4", "reverse", "natural"), ("r1e1", "reverse", "natural"), [("r1e4", "reverse"), ("r1e1", "reverse")]),
 ("T1", "area_dev_cmstep", ("r2e1", "reverse", "natural"), ("r1e1", "reverse", "natural"), [("r2e1", "reverse"), ("r1e1", "reverse")]),
 ("T2", "area_dev_cmstep", ("r4e1", "feedback", "corrected"), ("r1e1", "feedback", "corrected"), [("r4e1", "feedback"), ("r4e1", "corrected"), ("r1e1", "feedback"), ("r1e1", "corrected")]),
 ("T3", "area_dev_cmstep", ("r1e4", "reverse", "natural"), ("r1e1", "reverse", "natural"), [("r1e4", "reverse"), ("r1e1", "reverse")]),
 ("T4", "area_dev_cmstep", ("r1e4", "feedback", "corrected"), ("r1e1", "feedback", "corrected"), [("r1e4", "feedback"), ("r1e4", "corrected"), ("r1e1", "feedback"), ("r1e1", "corrected")]),
 ("T5", "max_dev_cm", ("r4e1", "reverse", "natural"), ("r1e1", "reverse", "natural"), [("r4e1", "reverse"), ("r1e1", "reverse")]),
 ("T6", "realign_t80", ("r4e1", "reverse", "natural"), ("r1e1", "reverse", "natural"), [("r4e1", "reverse"), ("r1e1", "reverse")]),
]
for name, metric, late, early, arms in SPEC:
    for label, rule in (("all", lambda t, s: True), ("zero_flip", lambda t, s: all(nflip(c, m, t, s) == 0 for c, m in arms))):
        ks = [(t, s) for t, s in keys if rule(t, s)]
        v = np.array([cv(*late, metric, t, s) - cv(*early, metric, t, s) for t, s in ks], float)
        if len(v) < 4: print(name, label, "n", len(v)); continue
        bm = boot(v, ks); per = {t: (int(sum(k[0] == t for k in ks)), round(float(np.mean([x for x, k in zip(v, ks) if k[0] == t])), 3)) for t in tasks}
        print(f"{name} {metric} {label}: n={len(v)} per_task={per} est={A.strat_mean(v, ks):.3f} "
              f"CI95=[{np.percentile(bm,2.5):.3f},{np.percentile(bm,97.5):.3f}] CI90=[{np.percentile(bm,5):.3f},{np.percentile(bm,95):.3f}] "
              f"p_signflip={A.signflip(v):.4f}")
