# Exploratory: noise floor comparison on units 0-39 (gen/null measured only there). Committed analyze_p1d functions only.
import sys, json, numpy as np
sys.path.insert(0, "/root/VLA/autovla_misalignment_poc/scripts/review_followup/p1d")
import analyze_p1d as A
EP = "/root/VLA/autovla_misalignment_poc/outputs/review_followup/P1D_dose_control/run_20261008/rollouts/main/episodes.jsonl"
C = {}
for line in open(EP):
    ep = json.loads(line); C[(ep["cond"], ep["task"], ep["seed"])] = A.compact(ep)
keys = sorted({(t, s) for (c, t, s) in C if c == "r1e1:gen"})
print("n units with gen:", len(keys))
def boot(v, ks):
    tk = np.array([k[0] for k in ks]); ms = []
    for t in sorted(set(tk)):
        ix = np.nonzero(tk == t)[0]
        ms.append(v[ix][np.random.default_rng(0).integers(0, len(ix), size=(A.B_REPS, len(ix)))].mean(1))
    return np.mean(ms, 0)
def cv(cfg, x, y, m, t, s): return A.pair(C[(f"{cfg}:{x}", t, s)], C[(f"{cfg}:{y}", t, s)])[m]
def rep(name, v, ks):
    bm = boot(v, ks); print(f"{name}: n={len(v)} est={A.strat_mean(v, ks):.1f} CI95=[{np.percentile(bm,2.5):.1f},{np.percentile(bm,97.5):.1f}] p={A.signflip(v):.4f}")
m = "area_dev_cmstep"
for cfg in ("r1e1", "r4e1"):
    rep(f"{cfg} gen-N", np.array([cv(cfg, "gen", "natural", m, t, s) for t, s in keys]), keys)
    rep(f"{cfg} R-N (units0-39)", np.array([cv(cfg, "reverse", "natural", m, t, s) for t, s in keys]), keys)
rep("gen DiD r4e1-r1e1", np.array([cv("r4e1", "gen", "natural", m, t, s) - cv("r1e1", "gen", "natural", m, t, s) for t, s in keys]), keys)
rep("H1 on units 0-39", np.array([cv("r4e1", "reverse", "natural", m, t, s) - cv("r1e1", "reverse", "natural", m, t, s) for t, s in keys]), keys)
rep("H1 minus gen DiD (units 0-39)", np.array([(cv("r4e1", "reverse", "natural", m, t, s) - cv("r1e1", "reverse", "natural", m, t, s)) - (cv("r4e1", "gen", "natural", m, t, s) - cv("r1e1", "gen", "natural", m, t, s)) for t, s in keys]), keys)
