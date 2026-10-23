#!/usr/bin/env python
"""P4 power / precision simulation for new independent physical states (SpatialVLA, SimplerEnv), planning only.

Planning data = P1-D (restore-verified, same intervention schedule 8/12/16/20 as proposed for P4) unit vectors of the
8 success outcomes {N,F,C,R} x {r1e4,r4e1} per task (planning_units_p1d.npz from p4_planning_data.py). Exp 35 enters
only through the stress scenario (its higher discordance). No confirmatory data exist; nothing here is a test of
the P4 hypotheses.

Generative model per simulated state (one task): draw a P1-D unit of that task with replacement (keeps the joint
structure of the 4 paired differences dR_r1e4, dC_r1e4, dR_r4e1, dC_r4e1), multiply the whole 4-vector by a random
sign (true mean exactly 0, discordance and cross-contrast correlation preserved), then impose the scenario:
  null        no change (delta = 0)
  small+      delta = +2.5 pp on every contrast in both tasks (concordant pairs turned +1 w.p. delta/(1-pi))
  small_opp   delta = +3 pp in coke, -3 pp in move_near (task x intervention = 6 pp, equal-task mean 0)
  het10       delta = 0 on average, but each state carries a +/-10 pp state-specific effect (random sign per state):
              extra discordance 0.10
  stress35    delta = 0, discordance raised to 0.35 on every contrast (exp-35-like upper bound)
  mid5        delta = +5 pp on every contrast (power must collapse for the 5 pp margin; shows the boundary)
Decision rule = the confirmatory one (stats_core.tost(method='signflip') on per-state totals, every state its own
cluster), evaluated with the large-G normal approximation of the sign-flip statistic
  p_lower = 1 - Phi( sum(d+m) / sqrt(sum (d+m)^2) ),  p_upper = Phi( sum(d-m) / sqrt(sum (d-m)^2) ),
which is what the Monte Carlo sign-flip converges to for G >= 100 (checked with --validate against stats_core).
Families: H8 = Holm over the 8 per-task equivalence tests (primary); H4 = Holm per task (4); raw = unadjusted.

Also: (a) task x intervention (coke - move_near) difference and +/-10 equivalence, (b) equal-task mean,
(c) the benchmark-only alternative for move_near: 10 fixed layouts, exact sign-flip TOST on 10 layout means
    with between-layout SD sigma_b (the only benchmark-valid move_near design, see protocol).

  source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
  nice -n 19 python p4_power_sim.py               # writes outputs/review_followup3/P4/power/
  nice -n 19 python p4_power_sim.py --validate    # normal approximation vs stats_core sign-flip (small check)
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = "/root/VLA/autovla_misalignment_poc"
PLAN = f"{ROOT}/outputs/review_followup3/P4/planning/planning_units_p1d.npz"
OUT = f"{ROOT}/outputs/review_followup3/P4/power"
KEYS = ["dR_r1e4", "dC_r1e4", "dR_r4e1", "dC_r4e1"]
TASKS = ["coke", "move_near"]
NS = [100, 150, 200, 250, 300, 350, 400, 500, 600, 800, 1000, 1200, 1500, 2000]
SIMS = 2000
SEED = 20261014
# throughput measured in P1-D main run (8 workers, RTX 5090 shared with an external process): block 1 1.83 ep/min,
# block 2 2.05 ep/min (analysis.md lines 30-33)
RATE_LO, RATE_HI = 1.83, 2.05
EP_PER_STATE = 8          # {N,F,C,R} x {r1e4,r4e1}
NULL_FRAC = 0.10          # null (no-op parity) arm on the first 10% of states per config: +0.2 episode/state
PILOT_STATES = 16         # per task, separate feasibility pilot (not confirmatory)
BUDGET_H = 48.0


def d_matrix(Y):
    # columns: r1e4 N F C R, r4e1 N F C R
    return np.column_stack([Y[:, 3] - Y[:, 0], Y[:, 2] - Y[:, 1], Y[:, 7] - Y[:, 4], Y[:, 6] - Y[:, 5]]).astype(float)


def draw(D, n, sims, rng, scen, task):
    idx = rng.integers(0, len(D), size=(sims, n))
    X = D[idx]                                                  # sims x n x 4
    X = X * rng.choice([-1.0, 1.0], size=(sims, n, 1))          # joint sign flip -> true mean 0
    pi = (D != 0).mean(0)                                       # planning discordance per contrast
    zero = X == 0
    u = rng.random(X.shape)
    if scen in ("small+", "small_opp", "mid5"):
        delta = {"small+": 0.025, "mid5": 0.05, "small_opp": 0.03 if task == "coke" else -0.03}[scen]
        q = abs(delta) / (1 - pi)
        X = np.where(zero & (u < q), np.sign(delta), X)
    elif scen == "het10":
        s = rng.choice([-1.0, 1.0], size=(sims, n, 1))
        q = 0.10 / (1 - pi)
        X = np.where(zero & (u < q), s * np.ones_like(X), X)
    elif scen == "stress35":
        q = np.clip((0.35 - pi) / (1 - pi), 0, 1)
        s = rng.choice([-1.0, 1.0], size=X.shape)
        X = np.where(zero & (u < q), s, X)
    return X


def sf_p(X, m):
    a, b = X + m, X - m
    with np.errstate(invalid="ignore", divide="ignore"):
        zl = a.sum(1) / np.sqrt((a ** 2).sum(1))
        zu = b.sum(1) / np.sqrt((b ** 2).sum(1))
    return np.maximum(norm.sf(zl), norm.cdf(zu))               # p_equiv per sim x contrast


def holm_reject(P, alpha=0.05):
    """P: sims x k -> boolean rejections under Holm."""
    sims, k = P.shape
    order = np.argsort(P, axis=1)
    Ps = np.take_along_axis(P, order, 1)
    thr = alpha / (k - np.arange(k))
    ok = np.cumprod(Ps < thr, axis=1).astype(bool)
    rej = np.zeros_like(ok)
    np.put_along_axis(rej, order, ok, 1)
    return rej


def run(D_by_task, rng):
    rows = []
    for scen in ("null", "small+", "small_opp", "het10", "stress35", "mid5"):
        for n in NS:
            Xs = {t: draw(D_by_task[t], n, SIMS, rng, scen, t) for t in TASKS}
            means = {t: Xs[t].mean(1) for t in TASKS}                      # sims x 4
            sds = {t: Xs[t].std(1, ddof=1) for t in TASKS}
            for m in (0.10, 0.075, 0.05):
                P = {t: sf_p(Xs[t], m) for t in TASKS}
                P8 = np.concatenate([P["coke"], P["move_near"]], 1)
                R8 = holm_reject(P8)
                R4 = {t: holm_reject(P[t]) for t in TASKS}
                Rraw = P8 < 0.05
                # equal-task mean (secondary, Holm over 4) and task x intervention (difference + equivalence, Holm 4)
                em = (means["coke"] + means["move_near"]) / 2
                se_em = 0.5 * np.sqrt(sds["coke"] ** 2 / n + sds["move_near"] ** 2 / n)
                pem = np.maximum(norm.sf((em + m) / se_em), norm.cdf((em - m) / se_em))
                Rem = holm_reject(pem)
                it = means["coke"] - means["move_near"]
                se_it = np.sqrt(sds["coke"] ** 2 / n + sds["move_near"] ** 2 / n)
                pit_diff = 2 * norm.sf(np.abs(it / se_it))
                pit_eq = np.maximum(norm.sf((it + m) / se_it), norm.cdf((it - m) / se_it))
                Rit_diff = holm_reject(pit_diff)
                Rit_eq = holm_reject(pit_eq)
                for ti, t in enumerate(TASKS):
                    for ki, k in enumerate(KEYS):
                        col = ti * 4 + ki
                        rows.append({
                            "scenario": scen, "n_states_per_task": n, "margin_pp": round(100 * m, 1), "task": t,
                            "contrast": k,
                            "power_H8": R8[:, col].mean(), "power_H4": R4[t][:, ki].mean(),
                            "power_raw": Rraw[:, col].mean(),
                            "ci95_halfwidth_pp_median": float(np.median(100 * 1.96 * sds[t][:, ki] / np.sqrt(n))),
                            "ci90_halfwidth_pp_median": float(np.median(100 * 1.645 * sds[t][:, ki] / np.sqrt(n))),
                            "discordance_mean": float((Xs[t][:, :, ki] != 0).mean()),
                            "true_mean_pp_sim": float(100 * means[t][:, ki].mean()),
                            "all8_H8": R8.all(1).mean(), "all4_task_H4": R4[t].all(1).mean(),
                            "eqmean_power_H4": Rem[:, ki].mean(),
                            "eqmean_ci95_halfwidth_pp": float(np.median(100 * 1.96 * se_em[:, ki])),
                            "inter_diff_power_H4": Rit_diff[:, ki].mean(),
                            "inter_equiv_power_H4": Rit_eq[:, ki].mean(),
                            "inter_ci95_halfwidth_pp": float(np.median(100 * 1.96 * se_it[:, ki])),
                        })
        print(f"[sim] {scen} done", flush=True)
    return pd.DataFrame(rows)


def layouts10(D_mn, rng, sims=2000):
    """Benchmark-only move_near: 10 fixed layouts, m units per layout (24 = full census of 6 instructions x 4 robot
    textures), layout effect ~ N(0, sigma_b) added as extra concordant->discordant conversion; exact sign-flip TOST on
    10 layout means (1024 sign patterns)."""
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=10)))   # 1024 x 10
    rows = []
    for sb in (0.0, 0.05, 0.10, 0.15):
        for mper in (12, 24):
            X = draw(D_mn, 10 * mper, sims, rng, "null", "move_near").reshape(sims, 10, mper, 4)
            lm = X.mean(2)                                              # sims x 10 x 4
            lm = lm + rng.normal(0, sb, size=(sims, 10, 1)) * np.ones((1, 1, 4))
            for m in (0.10, 0.05):
                P = np.zeros((sims, 4))
                for k in range(4):
                    a = lm[:, :, k] + m
                    b = lm[:, :, k] - m
                    Ta = a.sum(1); Tb = b.sum(1)
                    pa = ((np.abs(a)[:, None, :] * signs[None]).sum(2) >= Ta[:, None] - 1e-12).mean(1)
                    pb = ((np.abs(b)[:, None, :] * signs[None]).sum(2) <= Tb[:, None] + 1e-12).mean(1)
                    P[:, k] = np.maximum(pa, pb)
                R = P < 0.05
                for k in range(4):
                    sdl = lm[:, :, k].std(1, ddof=1)
                    rows.append({"sigma_between_pp": 100 * sb, "units_per_layout": mper, "margin_pp": 100 * m,
                                 "contrast": KEYS[k], "power_raw": R[:, k].mean(),
                                 "power_H4": holm_reject(P)[:, k].mean(),
                                 "ci95_halfwidth_pp_t9_median": float(np.median(100 * 2.262 * sdl / np.sqrt(10)))})
    return pd.DataFrame(rows)


def validate(D_by_task, rng):
    sys.path.insert(0, f"{ROOT}/scripts/review_followup2")
    import stats_core as sc
    out = []
    for n, m, scen in ((150, 0.10, "null"), (300, 0.10, "null"), (300, 0.05, "small+")):
        X = draw(D_by_task["move_near"], n, 60, rng, scen, "move_near")[:, :, 2]        # dR_r4e1
        pa = sf_p(X[:, :, None], m)[:, 0]
        for i in range(len(X)):
            r = sc.tost(X[i], np.ones(n), m, method="signflip", reps=20000, B=200)
            out.append({"n": n, "m": m, "scen": scen, "p_normal": float(pa[i]), "p_statscore": float(r["p_equiv"])})
    df = pd.DataFrame(out)
    df["abs_diff"] = (df.p_normal - df.p_statscore).abs()
    df["same_decision_05"] = (df.p_normal < 0.05) == (df.p_statscore < 0.05)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    z = np.load(PLAN, allow_pickle=True)
    D = {t: d_matrix(z[f"{t}_Y"]) for t in TASKS}
    rng = np.random.default_rng(SEED)
    if a.validate:
        v = validate(D, rng)
        v.to_csv(os.path.join(OUT, "validate_normal_vs_statscore.csv"), index=False)
        print(v.groupby(["n", "m", "scen"]).agg(maxdiff=("abs_diff", "max"), agree=("same_decision_05", "mean")))
        return
    df = run(D, rng)
    df.to_csv(os.path.join(OUT, "power_grid.csv"), index=False, float_format="%.4f")
    lay = layouts10(D["move_near"], rng)
    lay.to_csv(os.path.join(OUT, "power_movenear_10layouts.csv"), index=False, float_format="%.4f")

    # minimum n (states per task) reaching 80% power, per scenario x margin x contrast x family
    req = []
    for (scen, mg, t, k), g in df.groupby(["scenario", "margin_pp", "task", "contrast"]):
        g = g.sort_values("n_states_per_task")
        rec = {"scenario": scen, "margin_pp": mg, "task": t, "contrast": k}
        for fam in ("power_H8", "power_H4", "power_raw"):
            ok = g[g[fam] >= 0.80]
            rec[f"n80_{fam[6:]}"] = int(ok.n_states_per_task.iloc[0]) if len(ok) else f">{NS[-1]}"
        req.append(rec)
    req = pd.DataFrame(req)
    req.to_csv(os.path.join(OUT, "n_required_80.csv"), index=False)
    allp = df.groupby(["scenario", "margin_pp", "n_states_per_task"])[["all8_H8"]].first().reset_index()
    allp.to_csv(os.path.join(OUT, "power_all8.csv"), index=False, float_format="%.4f")

    budget = []
    for n in NS:
        eps = 2 * (n * EP_PER_STATE + int(np.ceil(NULL_FRAC * n)) * 2)
        pil = 2 * PILOT_STATES * (EP_PER_STATE + 2)
        budget.append({"n_states_per_task": n, "confirmatory_episodes": eps, "pilot_episodes": pil,
                       "gpu_h_at_2.05": round((eps + pil) / RATE_HI / 60, 1),
                       "gpu_h_at_1.83": round((eps + pil) / RATE_LO / 60, 1),
                       "within_48h_conservative": (eps + pil) / RATE_LO / 60 <= BUDGET_H})
    pd.DataFrame(budget).to_csv(os.path.join(OUT, "budget.csv"), index=False)
    json.dump({"sims": SIMS, "seed": SEED, "NS": NS, "rates": [RATE_LO, RATE_HI], "budget_h": BUDGET_H,
               "ep_per_state": EP_PER_STATE, "null_frac": NULL_FRAC, "pilot_states_per_task": PILOT_STATES,
               "planning_source": PLAN}, open(os.path.join(OUT, "sim_meta.json"), "w"), indent=1)
    print(req.to_string())
    print(pd.DataFrame(budget).to_string())
    print(lay.to_string())


if __name__ == "__main__":
    main()
