#!/usr/bin/env python
"""P4 power / precision simulation for NEW independent SimplerEnv variant states (SpatialVLA, closed loop). PLANNING ONLY.

CP2 (2026-10-14), regenerated (the stopped v1 power script was reviewed, not reused). No confirmatory data exist; the
existing P1-D units are used only as planning data (joint distribution of the 4 paired differences per task).

Design simulated (one task): n new physical states, 1 rollout per arm per state (SpatialVLA decoding is deterministic:
C5a/C5b reruns are bitwise identical, so repeated rollouts of the same state carry no information), arms
{N,F,C,R} x {r1e4,r4e1} = 8 episodes per state. Contrasts per task: dR_r1e4, dC_r1e4, dR_r4e1, dC_r4e1
(dR = R-N, dC = C-F). Every state is its own cluster (one unit per layout), so within-state correlation does not enter
the primary design; a k-instructions-per-layout alternative is evaluated with the ICC from planning_within_state.csv.

Generative model per simulated state: draw a P1-D unit of that task with replacement (keeps discordance and the
cross-contrast correlation), multiply the 4-vector by a random sign (true mean 0), then apply the scenario:
  true0       delta = 0 (named true0, not "null": pandas would read "null" as NaN)
  small_conc  delta = +2.5 pp on every contrast, both tasks (concordant pairs turned +1 w.p. delta/(1-p_d))
  small_opp   delta = +3 pp coke, -3 pp move_near (task x intervention 6 pp; equal-task mean 0)
  het_state   delta = 0 on average; each state carries a +/-10 pp state-specific effect (random sign per state)
  stress35    delta = 0, discordance raised to 0.35 on every contrast (exp-35-like upper bound)
  boundary5   delta = +5 pp on every contrast (true effect on the 5 pp margin: equivalence power must be ~alpha)
Decision rule = confirmatory TOST (alpha 0.05) with the state-level sign-flip statistic, large-n normal approximation
  p_lo = 1 - Phi(sum(d+m)/sqrt(sum((d+m)^2))), p_hi = Phi(sum(d-m)/sqrt(sum((d-m)^2))), p = max(p_lo, p_hi)
(--validate compares it with a Monte Carlo sign-flip at n = 120). Families: raw (per test), H4 = Holm within task
(4 tests, 'all 4 pass'), H8 = Holm over the 8 per-task tests ('all 8 pass', the primary family proposed in the protocol).
Also: 95% CI half-width (pp), task x intervention difference (coke - move_near) +/-10 pp TOST, equal-task mean.

Cost: 8 episodes / state / task + no-op/restore control arms on 5% of states (2 episodes each) + a separate
16-state-per-task feasibility pilot (not confirmatory). Rates are read from the run logs by stage3_cost_rates.py.

  source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
  nice -n 19 python p4_power_sim.py [--validate]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = "/root/VLA/autovla_misalignment_poc"
PLAN = f"{ROOT}/outputs/review_followup3/P4/planning"
OUT = f"{ROOT}/outputs/review_followup3/P4/power"
RATES = f"{ROOT}/outputs/review_followup3/STAGE3_COST_RATES.csv"
KEYS = ["dR_r1e4", "dC_r1e4", "dR_r4e1", "dC_r4e1"]
TASKS = ["coke", "move_near"]
NS = [100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 1000, 1200, 1500, 2000, 2500, 3000]
MARGINS = [0.05, 0.06, 0.075, 0.10]
SCEN = ["true0", "small_conc", "small_opp", "het_state", "stress35", "boundary5"]
SIMS = 2000
CHUNK = 250
SEED = 20261014
EP_PER_STATE = 8
CTRL_FRAC, CTRL_EP = 0.05, 2
PILOT_STATES = 16
BUDGET_H = 48.0
ALPHA = 0.05


def d_matrix(Y):
    # columns of Y: r1e4 N F C R, r4e1 N F C R  ->  dR_r1e4, dC_r1e4, dR_r4e1, dC_r4e1
    return np.column_stack([Y[:, 3] - Y[:, 0], Y[:, 2] - Y[:, 1], Y[:, 7] - Y[:, 4], Y[:, 6] - Y[:, 5]]).astype(float)


def draw(D, n, sims, rng, scen, task):
    X = D[rng.integers(0, len(D), size=(sims, n))]
    X = X * rng.choice([-1.0, 1.0], size=(sims, n, 1))
    pi = (D != 0).mean(0)
    zero = X == 0
    u = rng.random(X.shape)
    if scen in ("small_conc", "small_opp", "boundary5"):
        delta = {"small_conc": 0.025, "boundary5": 0.05, "small_opp": 0.03 if task == "coke" else -0.03}[scen]
        X = np.where(zero & (u < abs(delta) / (1 - pi)), np.sign(delta), X)
    elif scen == "het_state":
        s = rng.choice([-1.0, 1.0], size=(X.shape[0], n, 1))
        X = np.where(zero & (u < 0.10 / (1 - pi)), s * np.ones_like(X), X)
    elif scen == "stress35":
        q = np.clip((0.35 - pi) / (1 - pi), 0, 1)
        X = np.where(zero & (u < q), rng.choice([-1.0, 1.0], size=X.shape), X)
    return X


def tost_p(X, m):
    lo, hi = X + m, X - m
    with np.errstate(divide="ignore", invalid="ignore"):
        p_lo = 1 - norm.cdf(lo.sum(1) / np.sqrt((lo ** 2).sum(1)))
        p_hi = norm.cdf(hi.sum(1) / np.sqrt((hi ** 2).sum(1)))
    return np.maximum(p_lo, p_hi)                                   # sims x 4


def holm_all(P):
    """P: sims x m. True iff Holm (alpha) rejects every hypothesis."""
    m = P.shape[1]
    Ps = np.sort(P, axis=1)
    thr = ALPHA / (m - np.arange(m))
    return (Ps <= thr).all(1)


def diff_tost(Xa, Xb, m):
    """independent-sample z TOST of mean(a) - mean(b), per contrast."""
    na, nb = Xa.shape[1], Xb.shape[1]
    diff = Xa.mean(1) - Xb.mean(1)
    se = np.sqrt(Xa.var(1, ddof=1) / na + Xb.var(1, ddof=1) / nb)
    p = np.maximum(1 - norm.cdf((diff + m) / se), norm.cdf((diff - m) / se))
    return p


def simulate(Ds, rng):
    rows = []
    for scen in SCEN:
        for n in NS:
            acc = {}
            for c0 in range(0, SIMS, CHUNK):
                sims = min(CHUNK, SIMS - c0)
                X = {t: draw(Ds[t], n, sims, rng, scen, t) for t in TASKS}
                for m in MARGINS:
                    P = {t: tost_p(X[t], m) for t in TASKS}
                    allP = np.concatenate([P["coke"], P["move_near"]], axis=1)
                    acc.setdefault(("raw", m), []).append((allP <= ALPHA).mean(0))
                    acc.setdefault(("H4", m), []).append(np.array([holm_all(P[t]).mean() for t in TASKS]))
                    acc.setdefault(("H8", m), []).append(np.array([holm_all(allP).mean()]))
                    if m == 0.10:
                        acc.setdefault(("taskdiff10", m), []).append((diff_tost(X["coke"], X["move_near"], m) <= ALPHA).mean(0))
                        eq = (X["coke"] + X["move_near"]) / 2.0
                        acc.setdefault(("eqmean10", m), []).append((tost_p(eq, m) <= ALPHA).mean(0))
                hw = {t: 100 * 1.96 * X[t].std(1, ddof=1).mean(0) / np.sqrt(n) for t in TASKS}
                acc.setdefault(("hw", 0), []).append(np.concatenate([hw["coke"], hw["move_near"]]))
                acc.setdefault(("bias", 0), []).append(np.concatenate([100 * X[t].mean(1).mean(0) for t in TASKS]))
            w = None
            for (kind, m), lst in acc.items():
                v = np.average(np.vstack(lst), axis=0, weights=[min(CHUNK, SIMS - c) for c in range(0, SIMS, CHUNK)])
                if kind in ("raw", "hw", "bias"):
                    for ti, t in enumerate(TASKS):
                        for ki, k in enumerate(KEYS):
                            rows.append({"scenario": scen, "n_states_per_task": n, "margin_pp": 100 * m,
                                         "metric": {"raw": "power_raw", "hw": "ci95_halfwidth_pp",
                                                    "bias": "mean_effect_pp"}[kind],
                                         "task": t, "contrast": k, "value": v[ti * 4 + ki]})
                elif kind == "H4":
                    for ti, t in enumerate(TASKS):
                        rows.append({"scenario": scen, "n_states_per_task": n, "margin_pp": 100 * m,
                                     "metric": "power_all4_holm_within_task", "task": t, "contrast": "all4",
                                     "value": v[ti]})
                elif kind == "H8":
                    rows.append({"scenario": scen, "n_states_per_task": n, "margin_pp": 100 * m,
                                 "metric": "power_all8_holm8", "task": "both", "contrast": "all8", "value": v[0]})
                elif kind == "taskdiff10":
                    for ki, k in enumerate(KEYS):
                        rows.append({"scenario": scen, "n_states_per_task": n, "margin_pp": 10.0,
                                     "metric": "power_taskdiff_equiv_raw", "task": "coke-move_near", "contrast": k,
                                     "value": v[ki]})
                elif kind == "eqmean10":
                    for ki, k in enumerate(KEYS):
                        rows.append({"scenario": scen, "n_states_per_task": n, "margin_pp": 10.0,
                                     "metric": "power_equal_task_mean_equiv_raw", "task": "equal-task mean",
                                     "contrast": k, "value": v[ki]})
            print(f"[sim] {scen} n={n} done", flush=True)
    return pd.DataFrame(rows)


def required_n(df):
    out = []
    for scen in SCEN:
        for m in MARGINS:
            s = df[(df.scenario == scen) & (np.isclose(df.margin_pp, 100 * m))]
            crit = {
                "min per-test power (raw) >= 0.8": s[s.metric == "power_raw"].groupby("n_states_per_task").value.min(),
                "both tasks all-4 pass, Holm m=4 within task >= 0.8":
                    s[s.metric == "power_all4_holm_within_task"].groupby("n_states_per_task").value.min(),
                "all 8 pass, Holm m=8 >= 0.8": s[s.metric == "power_all8_holm8"].groupby("n_states_per_task").value.min(),
            }
            for name, ser in crit.items():
                ok = ser[ser >= 0.8]
                out.append({"scenario": scen, "margin_pp": 100 * m, "criterion": name,
                            "n_states_per_task_required": int(ok.index.min()) if len(ok) else np.nan,
                            "power_at_max_n": float(ser.loc[max(NS)]),
                            "note": "" if len(ok) else f"> {max(NS)} states per task"})
    return pd.DataFrame(out)


def episodes(n_states):
    per_task = n_states * EP_PER_STATE + int(np.ceil(CTRL_FRAC * n_states)) * CTRL_EP
    return 2 * per_task + 2 * PILOT_STATES * EP_PER_STATE


def cost_table(req, rates):
    rows = []
    for _, r in req.iterrows():
        if np.isnan(r.n_states_per_task_required):
            continue
        n = int(r.n_states_per_task_required)
        ep = episodes(n)
        row = {**r.to_dict(), "episodes_total": ep}
        for _, rr in rates.iterrows():
            row[f"gpu_h_at_{rr.rate_id}"] = ep / rr.rate_per_min / 60.0
        rows.append(row)
    return pd.DataFrame(rows)


def budget_table(df, rates):
    rows = []
    for _, rr in rates.iterrows():
        ep_budget = BUDGET_H * 60 * rr.rate_per_min
        # largest grid n whose total episodes fit in the budget
        fit = [n for n in NS if episodes(n) <= ep_budget]
        n_b = max(fit) if fit else 0
        for scen in ("true0", "het_state", "stress35"):
            for m in MARGINS:
                s = df[(df.scenario == scen) & np.isclose(df.margin_pp, 100 * m) & (df.n_states_per_task == n_b)]
                rows.append({"rate_id": rr.rate_id, "rate_per_min": rr.rate_per_min, "budget_gpu_h": BUDGET_H,
                             "episodes_in_budget": int(ep_budget), "n_states_per_task_in_budget": n_b,
                             "scenario": scen, "margin_pp": 100 * m,
                             "min_per_test_power_raw": s[s.metric == "power_raw"].value.min() if len(s) else np.nan,
                             "power_all8_holm8": s[s.metric == "power_all8_holm8"].value.min() if len(s) else np.nan})
    return pd.DataFrame(rows)


def validate(Ds, rng):
    """normal-approximation TOST vs Monte Carlo sign-flip TOST (5,000 flips) at n = 120, null scenario."""
    out = []
    for t in TASKS:
        X = draw(Ds[t], 120, 100, rng, "true0", t)
        pa = tost_p(X, 0.10)
        flips = rng.choice([-1.0, 1.0], size=(5000, 120))
        pm = np.zeros_like(pa)
        for s in range(X.shape[0]):
            for k in range(4):
                lo, hi = X[s, :, k] + 0.10, X[s, :, k] - 0.10
                p_lo = ((flips * np.abs(lo)).sum(1) >= lo.sum()).mean()
                p_hi = ((flips * np.abs(hi)).sum(1) <= hi.sum()).mean()
                pm[s, k] = max(p_lo, p_hi)
        out.append({"task": t, "max_abs_diff_p": float(np.abs(pa - pm).max()),
                    "decision_agreement": float(((pa <= ALPHA) == (pm <= ALPHA)).mean())})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    z = np.load(os.path.join(PLAN, "planning_units_p1d.npz"), allow_pickle=True)
    Ds = {t: d_matrix(z[f"{t}_Y"]) for t in TASKS}
    rng = np.random.default_rng(SEED)
    if a.validate:
        v = validate(Ds, rng)
        v.to_csv(os.path.join(OUT, "power_validation.csv"), index=False)
        print(v)
        return
    rates = pd.read_csv(RATES)
    rates = rates[rates.model == "SpatialVLA"][["rate_id", "rate_per_min"]]
    df = simulate(Ds, rng)
    df.to_csv(os.path.join(OUT, "power_grid.csv"), index=False, float_format="%.4f")
    req = required_n(df)
    req.to_csv(os.path.join(OUT, "power_required_n.csv"), index=False, float_format="%.4f")
    ct = cost_table(req, rates)
    ct.to_csv(os.path.join(OUT, "power_cost.csv"), index=False, float_format="%.2f")
    bt = budget_table(df, rates)
    bt.to_csv(os.path.join(OUT, "power_budget48h.csv"), index=False, float_format="%.4f")
    # analytic rule of thumb from the prompt, for comparison only
    pl = pd.read_csv(os.path.join(PLAN, "planning_params.csv"))
    pl = pl[pl.experiment == "P1D"].copy()
    for m in (0.10, 0.05):
        pl[f"n_approx_m{int(100 * m)}"] = np.ceil((1.645 + 1.282) ** 2 * pl.discordance / m ** 2)
    pl.to_csv(os.path.join(OUT, "power_analytic_approx.csv"), index=False, float_format="%.4f")
    json.dump({"seed": SEED, "sims": SIMS, "ns": NS, "margins": MARGINS, "scenarios": SCEN, "ep_per_state": EP_PER_STATE,
               "ctrl_frac": CTRL_FRAC, "ctrl_ep": CTRL_EP, "pilot_states_per_task": PILOT_STATES, "budget_h": BUDGET_H,
               "alpha": ALPHA, "planning_data": "P1-D units (review_followup2 B episodes.csv)"},
              open(os.path.join(OUT, "power_meta.json"), "w"), indent=1)
    print(req.to_string())
    print(ct.to_string())
    print(bt.to_string())


if __name__ == "__main__":
    main()
