#!/usr/bin/env python
"""
Read-only cross-check of the navhard (exp 25-27, 29, 30) summary numbers against the per-group score files.
No new statistics: only group-score means, group counts (same EPS = 1e-9 as analyze_navhard_conditions.py),
derived ratios of already-reported differences, and token-level counts from candidate_oracle/half*/oracle/token_scores.npy.
Writes ../navhard_verify.json (next to this scripts/ dir).
"""
import json
import os

import numpy as np

POC = "/root/VLA/autovla_misalignment_poc/outputs"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "navhard_verify.json")
EPS = 1e-9
SUBSET = {"half1": f"{POC}/navhard_eval/subset/groups.json", "half2": f"{POC}/navhard_full_validation/half2/subset/groups.json"}
DIRS = [f"{POC}/navhard_full_validation/{{h}}/group_scores", f"{POC}/safety_filter_ablation/{{h}}/group_scores",
        f"{POC}/safety_filter_components/{{h}}/group_scores", f"{POC}/candidate_oracle/{{h}}/group_scores"]
CONDS = {"normal": ["normal"], "ranksum": ["ranksum"], "max_loglik": ["max_loglik"], "comp_collision": ["comp_collision"],
         "comp_dac": ["comp_dac"], "filter_only": ["filter_only"],
         "filter_random": ["filter_random_s0", "filter_random_s1", "filter_random_s2"],
         "F1": ["F1"], "F1_maxll": ["F1_maxll"], "filter_ranksum_natfb": ["filter_ranksum_natfb"],
         "filter_maxll_natfb": ["filter_maxll_natfb"], "oracle16": ["oracle16"], "oracle17": ["oracle17"]}
CONDS.update({f"cand{k:02d}": [f"cand{k:02d}"] for k in range(1, 17)})


def load(h, name):
    for d in DIRS:
        p = os.path.join(d.format(h=h), f"g_{name}.json")
        if os.path.exists(p):
            return {x["orig_token"]: x["group_score"] for x in json.load(open(p))}, p
    raise FileNotFoundError(name)


res = {"groups": {}, "cond": {}}
G = {}
for h, sp in SUBSET.items():
    g = json.load(open(sp))
    G[h] = {x["orig"]: x["log"] for x in g}
    res["groups"][h] = {"n_groups": len(G[h]), "n_logs": len(set(G[h].values()))}
res["groups"]["pooled"] = {"n_groups": sum(len(v) for v in G.values()),
                           "n_logs": len(set().union(*[set(v.values()) for v in G.values()]))}

S = {}
for c, files in CONDS.items():
    S[c] = {}
    src = []
    for h in SUBSET:
        parts = []
        for f in files:
            d, p = load(h, f); parts.append(d); src.append(p)
        S[c].update({(h, t): float(np.mean([pp[t] for pp in parts])) for t in parts[0]})
    res["cond"][c] = {"sources": src}
for c in CONDS:
    for scope in ("half1", "half2", "pooled"):
        keys = [k for k in S["normal"] if scope == "pooled" or k[0] == scope]
        a = np.array([S[c][k] for k in keys]); b = np.array([S["normal"][k] for k in keys])
        d = a - b
        res["cond"][c][scope] = {"n": len(keys), "epdms": float(a.mean()), "diff_vs_normal": float(d.mean()),
                                 "groups_better_worse_tied": [int((d > EPS).sum()), int((d < -EPS).sum()), int((np.abs(d) <= EPS).sum())]}

# derived ratios from the reported pooled contrasts (exact values from the analysis JSONs)
abl = json.load(open(f"{POC}/safety_filter_ablation/ablation.json"))["full"]["contrasts"]
cmp_ = json.load(open(f"{POC}/safety_filter_components/components.json"))["full"]["contrasts"]
orc = json.load(open(f"{POC}/candidate_oracle/oracle_comparison.json"))["full"]["contrasts"]
e = lambda D, k: D[k]["epdms"]["diff"]
F1 = e(abl, "filter_ranksum - normal"); F1m = e(abl, "filter_maxll - normal"); FO = e(abl, "filter_only - normal")
R = e(abl, "filter_ranksum - filter_only"); M = e(abl, "filter_maxll - filter_only")
C1 = e(cmp_, "collision_only - no_filter"); D1 = e(cmp_, "dac_only - no_filter"); CD = e(cmp_, "collision_dac - no_filter")
D_add = e(cmp_, "collision_dac - collision_only"); C_add = e(cmp_, "collision_dac - dac_only")
O17 = e(orc, "oracle17 - no_selection"); gap = e(orc, "oracle17 - filter_maxll"); gapr = e(orc, "oracle17 - filter_ranksum")
sh_da = 0.5 * (D1 + D_add); sh_c = 0.5 * (C1 + C_add)
res["derived"] = {
    "F1_total": F1, "F1maxll_total": F1m, "filter_only_gain": FO, "rank_marginal": R, "maxll_marginal": M,
    "safety_share_of_F1 = filter_only/F1": FO / F1, "safety_share_rounded_inputs = 0.105/0.118": 0.105 / 0.118,
    "rank_share_of_F1 = rank_marginal/F1": R / F1, "maxll_share_of_F1maxll = maxll_marginal/F1maxll": M / F1m,
    "safety_share_of_F1maxll = filter_only/F1maxll": FO / F1m,
    "check F1 == filter_only + rank_marginal": FO + R - F1,
    "shapley_DA = 0.5*(dac_only-none + both-collision_only)": sh_da, "shapley_collision = 0.5*(collision_only-none + both-dac_only)": sh_c,
    "shapley_DA_share = shapley_DA/both": sh_da / CD, "shapley_collision_share": sh_c / CD, "both_minus_none": CD,
    "interaction = both - dac_only - collision_only (all vs none)": CD - D1 - C1,
    "DA_marginal_share = (dac_only-none)/both": D1 / CD, "DA_marginal_share_of_F1 = (dac_only-none)/F1": D1 / F1,
    "oracle17_gain": O17, "deployed_fraction_maxll = (F1maxll)/oracle17_gain": F1m / O17,
    "deployed_fraction_F1 = F1/oracle17_gain": F1 / O17, "filter_only_fraction = filter_only/oracle17_gain": FO / O17,
    "remaining_headroom_vs_maxll": gap, "remaining_headroom_vs_F1": gapr,
    "check F1maxll + gap == oracle17_gain": F1m + gap - O17,
}
# score decomposition: multiplicative share per condition/stage
sd = json.load(open(f"{POC}/safety_filter_components/score_decomposition.json"))["conds"]
res["score_decomposition_mult_share"] = {f"{c}:{st}": sd[c][st]["multiplicative"]["mean"] / sd[c][st]["total"]["mean"]
                                         for c in sd for st in ("stage1", "stage2") if sd[c][st]["total"]["mean"] != 0}
res["score_decomposition_progress_term"] = {f"{c}:{st}": sd[c][st]["ego_progress"]["mean"] for c in sd for st in ("stage1", "stage2")}
res["plan_reach"] = json.load(open(f"{POC}/safety_filter_components/plan_reach.json"))

# token-level oracle matrices
tok = {}
allM = []
for h in SUBSET:
    Mx = np.load(f"{POC}/candidate_oracle/{h}/oracle/token_scores.npy"); allM.append(Mx)
    tok[h] = {"n_tokens": int(Mx.shape[0]), "n_cand": int(Mx.shape[1]), "frac_all17_zero": float((Mx.max(1) == 0).mean()),
              "n_all17_zero": int((Mx.max(1) == 0).sum()), "frac_natural_zero": float((Mx[:, 0] == 0).mean()),
              "mean_natural_token": float(Mx[:, 0].mean()), "mean_max17_token": float(Mx.max(1).mean()),
              "mean_max16_token": float(Mx[:, 1:].max(1).mean()), "oracle17_is_natural": float((Mx.argmax(1) == 0).mean())}
Mx = np.concatenate(allM)
tok["pooled"] = {"n_tokens": int(Mx.shape[0]), "frac_all17_zero": float((Mx.max(1) == 0).mean()), "n_all17_zero": int((Mx.max(1) == 0).sum()),
                 "frac_natural_zero": float((Mx[:, 0] == 0).mean()), "mean_natural_token": float(Mx[:, 0].mean()),
                 "mean_max17_token": float(Mx.max(1).mean()), "mean_max16_token": float(Mx[:, 1:].max(1).mean()),
                 "oracle17_is_natural": float((Mx.argmax(1) == 0).mean())}
res["oracle_tokens"] = tok

# cross-check means against the analysis JSONs
full = json.load(open(f"{POC}/navhard_full_validation/navhard_full_comparison.json"))
ablm = json.load(open(f"{POC}/safety_filter_ablation/ablation.json"))
cmpm = json.load(open(f"{POC}/safety_filter_components/components.json"))
orcm = json.load(open(f"{POC}/candidate_oracle/oracle_comparison.json"))
chk = {}
for c, (src, key) in {"normal": (full["pooled"]["rules"], "normal"), "ranksum": (full["pooled"]["rules"], "ranksum"),
                      "max_loglik": (full["pooled"]["rules"], "max_loglik"), "F1": (full["pooled"]["rules"], "F1"),
                      "F1_maxll": (full["pooled"]["rules"], "F1_maxll")}.items():
    chk[c] = src[key]["epdms"]["mean"] - res["cond"][c]["pooled"]["epdms"]
for c, k in {"filter_only": "filter_only", "filter_random": "filter_random"}.items():
    chk[c] = ablm["full"]["means"][k]["epdms"] - res["cond"][c]["pooled"]["epdms"]
for c, k in {"comp_collision": "collision_only", "comp_dac": "dac_only"}.items():
    chk[c] = cmpm["full"]["means"][k]["epdms"] - res["cond"][c]["pooled"]["epdms"]
for c in ("oracle16", "oracle17"):
    chk[c] = orcm["full"]["means"][c]["epdms"] - res["cond"][c]["pooled"]["epdms"]
res["json_minus_recomputed_mean"] = chk
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("groups", "derived", "oracle_tokens", "json_minus_recomputed_mean", "score_decomposition_mult_share")}, indent=1))
for c in CONDS:
    p = res["cond"][c]["pooled"]; print(c, p["n"], round(p["epdms"], 4), round(p["diff_vs_normal"], 4), p["groups_better_worse_tied"])
