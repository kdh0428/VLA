#!/usr/bin/env python
"""Author query Q3 (2026-10-19): FINAL_PAPER_NUMBERS.csv / .json — one row per (claim, cohort, estimand, metric, weighting,
cluster axis, family); values copied from stored outputs (no new estimation except the derived window numerator /
denominator, flagged 'derived'). CI columns follow the Q1 recommendation: cluster-robust t where approximate coverage was
checked or the structure matches; NR where no interval was validated. CI bounds are written as Python floats.

  nice -n 19 /root/miniforge3/envs/autovla/bin/python q3_final_numbers.py
"""
import hashlib
import json
import math

import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
O = f"{POC}/outputs"
OUT = f"{O}/manuscript_author_queries_20261019"
AV = ("AutoVLA (Qwen2.5-VL-3B)", "driving", "open-loop 5 s plan, NAVSIM navtest (offline replay)")
SV = ("SpatialVLA-4B", "manipulation", "closed loop, SimplerEnv Google Robot")
H = {}


def sha(p):
    if p not in H:
        H[p] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return H[p]


def f(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def ci(s):
    if isinstance(s, str) and s.strip().startswith("["):
        a = [f(v) for v in s.strip("[]").split(",")]
        return a if all(v is not None for v in a) else [None, None]
    if isinstance(s, (list, tuple)):
        return [f(s[0]), f(s[1])]
    return [None, None]


def add(rows, **k):
    base = dict.fromkeys(["claim_id", "model", "domain", "evaluation_mode", "contrast_definition_and_sign", "cohort", "estimand",
                          "outcome", "estimate", "unit", "ci_level", "ci_low", "ci_high", "ci_method", "ci_use_and_limitations",
                          "p_raw", "p_adjusted", "test", "null_assumptions", "family", "alpha", "n_units", "n_scenes", "n_drives",
                          "n_components", "weighting", "source_file", "source_row_id", "source_hash", "status"])
    base.update(k)
    if base["source_file"] and not str(base["source_file"]).startswith("derived"):
        base["source_hash"] = sha(base["source_file"])
    rows.append(base)


SIGNFLIP = ("exact two-sided drive sign-flip at 0 (enumeration of all 2^G sign patterns)",
            "drive totals independent and sign-symmetric under H0 (not an assumption-free test of mean = 0)")
SEV = "severe trajectory outcome = A- AND FDE5 > 3 m at the 5 s (10-pose) plan; pp of units, scene-weighted"


def main():
    rows = []
    # ---------------------------------------------------------------- repeated Recent-GT / Reverse-N (P0-A)
    pop_f = f"{O}/review_followup3/P0/POPULATION_EFFECTS.csv"
    pop = pd.read_csv(pop_f).set_index("row_id")
    aud = pd.read_csv(f"{O}/review_followup5/A_STATS/AUTOVLA_CI_AUDIT.csv")
    aud = aud[aud.source.str.contains("POPULATION")].set_index("row")
    defs = {"RN": "Recent-GT - Normal (repeated sliding GT correction of the latest context position); negative = reduction",
            "RV": "Reverse-N - Full GT-history (separate Normal-branch token injected into a GT context); positive = restoration",
            "DM": "direction-wrong - magnitude-wrong last-token error; positive = direction more harmful"}
    for rid, cohort, est, ncomp in (("P0A.DEV.RN.Am", "dev mechanism set, A- stratum", "A- (failure-selected)", 8),
                                    ("P0A.DEV.RV.Am", "dev mechanism set, A- stratum", "A- (failure-selected)", 8),
                                    ("P0A.DEV.DM.Am", "dev mechanism set, A- stratum", "A- (failure-selected)", 8),
                                    ("P0A.P1C.RN.F", "P1-C new drives, failure stratum F", "F", 6),
                                    ("P0A.P1C.RV.F", "P1-C new drives, failure stratum F", "F", 6),
                                    ("P0A.P1C.DM.F", "P1-C new drives, failure stratum F", "F", 6),
                                    ("P0A.P1C.RN.P", "P1-C evaluation frame (2,191 eligible scenes; 353 sampled, design-weighted)", "P (Hajek)", 6),
                                    ("P0A.P1C.RV.P", "P1-C evaluation frame (2,191 eligible scenes; 353 sampled, design-weighted)", "P (Hajek)", 6),
                                    ("P0A.P1C.DM.P", "P1-C evaluation frame (2,191 eligible scenes; 353 sampled, design-weighted)", "P (Hajek)", 6)):
        r, a = pop.loc[rid], aud.loc[rid]
        c = rid.split(".")[2]
        nc = ci(a.new_ci95)
        validated = nc[0] is not None
        add(rows, claim_id=f"REP.{rid}", model=AV[0], domain=AV[1], evaluation_mode=AV[2], contrast_definition_and_sign=defs[c],
            cohort=cohort, estimand=est, outcome=SEV, estimate=f(r.theta) * 100, unit="pp", ci_level=0.95 if validated else None,
            ci_low=nc[0] * 100 if validated else None, ci_high=nc[1] * 100 if validated else None,
            ci_method="cluster-robust t (G-1 df), drive clusters" if validated else "NR",
            ci_use_and_limitations=("approximate: simulated coverage 0.93-0.99 in contrast-specific scenarios (Q1); "
                                    "old drive bootstrap reported for reference only" if validated and c in ("RN", "RV")
                                    else "approximate (same structure as the Q1 checks; this contrast itself not simulated)" if validated
                                    else "no interval validated for this weighted structure; report estimate and p only"),
            p_raw=f(r.p_signflip_raw), p_adjusted=f(r.p_holm), test=SIGNFLIP[0], null_assumptions=SIGNFLIP[1],
            family=f"{r.family_id} (m = {r.family_m})", alpha=f(r.family_alpha), n_units=int(r.n_units), n_scenes=int(r.n_scenes),
            n_drives=int(r.n_drives), n_components=ncomp, weighting=r.weighting, source_file=pop_f, source_row_id=rid,
            status="final (A_STATS interval rule)")
    # ---------------------------------------------------------------- B2 matched prefix
    b2c_f = f"{O}/review_followup5/A_B2_CLUSTER/B2_CLUSTER_RESULTS.csv"
    b2c = pd.read_csv(b2c_f)
    mpr_f = f"{O}/review_followup4/B/B2/p1c_analysis/MATCHED_PREFIX_RESULTS.csv"
    mpr = pd.read_csv(mpr_f)
    bdef = {"C_GT": "GT_NSRC - GT_REF: frozen Normal-source token vs GT in the latest context position, behind a GT older position; positive = harm of the Normal-source value",
            "C_NAT": "NAT_NSRC - NAT_REF: same, behind the model's own natural older position",
            "INT": "C_GT - C_NAT (difference of two paired contrasts)",
            "C_PROV": "GT_CSRC - GT_NSRC: corrected-branch source vs Normal source (value difference, not pure provenance)"}
    for _, r in b2c[b2c.family_member].iterrows():
        cl = r.cluster_axis
        lo = hi = None; meth = "NR"; use = ""
        if cl == "drive" and r.estimand == "F":
            m = mpr[(mpr.n_old == 1) & (mpr.metric == "severe") & (mpr.contrast == r.contrast) & (mpr.subset == "all") &
                    (mpr.estimand == "F") & (mpr.cluster == "drive")].iloc[0]
            lo, hi = ci(m.clt_ci95); meth = "cluster-robust t (G-1 df), drive clusters"
            use = ("approximate: simulated coverage 0.93-0.99 (Q1, C_NAT)" if r.contrast == "C_NAT" else
                   "approximate sensitivity: joint-cell covariance of the two paired contrasts not modelled" if r.contrast == "INT" else
                   "approximate (same structure as the Q1 C_NAT check; not itself simulated)")
        elif cl == "component":
            use = "6 components: no interval; Holm floor 0.125 makes significance impossible at this level"
        else:
            use = "no validated interval for this structure (P weighted / day-vehicle); report estimate and p"
        add(rows, claim_id=f"B2.{r.contrast}.{r.estimand}.{cl}", model=AV[0], domain=AV[1], evaluation_mode=AV[2],
            contrast_definition_and_sign=bdef[r.contrast] + "; single retained edit at anchor a = t*+3, older factor = one position (t*+1); GT-conditioned diagnostic",
            cohort="P1-C B2 cohort (existing drives re-run)", estimand="F (failure stratum)" if r.estimand == "F" else "E4 (t* <= 4 eligible domain, Hajek)",
            outcome=SEV, estimate=f(r.theta) * 100, unit="pp", ci_level=0.95 if lo is not None else None,
            ci_low=lo * 100 if lo is not None else None, ci_high=hi * 100 if hi is not None else None, ci_method=meth,
            ci_use_and_limitations=use, p_raw=f(r.p_signflip_exact), p_adjusted=f(r.holm_p),
            test=SIGNFLIP[0].replace("drive", cl.replace("_", "-")), null_assumptions=SIGNFLIP[1].replace("drive", cl.replace("_", "-")),
            family=r.family, alpha=f(r.alpha), n_units=int(r.n_units), n_scenes=int(r.n_scenes),
            n_drives=18 if r.estimand == "F" else 24, n_components=int(r.G) if cl == "component" else 6, weighting="scene-weighted (E4: recomputed design weights)",
            source_file=b2c_f, source_row_id=f"{r.contrast}|{r.estimand}|{cl}", status="final")
    # ---------------------------------------------------------------- B_NUMERIC (GT-free endpoint excess)
    bn_f = f"{O}/review_followup5/B_NUMERIC/B_NUMERIC_RESULTS.csv"
    bn = pd.read_csv(bn_f)
    nm = {"Geo": "coordinate accumulation (Geo; privileged, uses the reference future)", "CM": "constant motion (body-frame SE(2) repeat incl. yaw)",
          "CV0": "zero-turn constant velocity", "CTRV-K3": "constant turn rate and velocity (last 3 steps)", "CTRA-2": "constant turn rate and acceleration",
          "E_prefix": "learned non-AR prefix extrapolator (3-seed mean)", "E_prefix_ego": "learned extrapolator, prefix + ego (secondary)"}
    for _, r in bn.iterrows():
        q = r.quantity
        if not (q.startswith("E_") or q == "Delta"):
            continue
        b = q[2:] if q.startswith("E_") else None
        grp = {"A-": "A- (failure-selected; 52 scenes)", "A+": "A+ (156 selected non-failure scenes)", "all": "all 208 selected mechanism scenes (not the natural set)"}[r.group]
        primary = isinstance(r.family, str) and r.family.startswith("primary")
        add(rows, claim_id=f"GTFREE.{r.setting}.{r.group}.{q}", model=AV[0], domain=AV[1], evaluation_mode=AV[2],
            contrast_definition_and_sign=(f"endpoint response D_model - D_baseline for {nm[b]}; positive = model moves more than the baseline"
                                         if b else f"setting change of the excess, {r.setting} - S0 (paired by unit; identical for all baselines)"),
            cohort="P1-A family G (GT-free perturbation at t*)", estimand=grp,
            outcome="endpoint response = ||X_OC(5 s) - X_U(5 s)|| (m), not FDE", estimate=f(r.est_scene), unit="m", ci_level=0.95,
            ci_low=f(r.ci95_lo), ci_high=f(r.ci95_hi), ci_method="drive-cluster percentile bootstrap (B = 10,000)",
            ci_use_and_limitations="continuous metric; bootstrap coverage not checked for this metric (Q1 covered binary contrasts); inference by sign-flip p (+ Holm for the primary family)",
            p_raw=f(r.p_signflip), p_adjusted=f(r.p_holm) if primary else None, test=SIGNFLIP[0], null_assumptions=SIGNFLIP[1],
            family=r.family if primary else ("secondary" if b else "setting change (descriptive)"), alpha=0.05 if primary else None,
            n_units=int(r.n_units), n_scenes=int(r.n_scenes), n_drives=int(r.n_drives), n_components=None,
            weighting="scene-weighted", source_file=bn_f, source_row_id=f"{r.setting}|{r.group}|{q}",
            status=f"final; setting {r.setting} ({'original computation' if r.setting == 'S0' else 'eager attention' if r.setting == 'SE' else 'fp32 arithmetic'})")
    # ---------------------------------------------------------------- retained three-position GT window (~90%)
    wf = f"{O}/review_followup3/P0/WINDOW_WEIGHT_AUDIT.csv"
    a2 = json.load(open(f"{O}/review_followup4/A2_A3/a2_checks.json"))["ninety_percent_definition"]
    lv = a2["scene_weighted_levels"]
    add(rows, claim_id="WINDOW.win3.fraction", model=AV[0], domain=AV[1], evaluation_mode=AV[2],
        contrast_definition_and_sign="(Normal - win3) / (Normal - GT-history): three fixed context positions replaced with GT and RETAINED in all later contexts (no release)",
        cohort="dev A- stratum (R9 recovery run of exp 9)", estimand="A-", outcome=SEV + " (ratio of differences)",
        estimate=0.9005216436350325 * 100, unit="% of the full-history correction effect", ci_level=0.95, ci_low=81.21527990006487,
        ci_high=98.92578825072238, ci_method="drive-cluster bootstrap, numerator and denominator resampled jointly",
        ci_use_and_limitations="bootstrap of a ratio; coverage not checked; 13 drives",
        p_raw=None, p_adjusted=None, test="none", null_assumptions="", family=None, alpha=None, n_units=365, n_scenes=52, n_drives=13,
        n_components=8, weighting="scene means first, scenes equal, ratio of sums", source_file=wf,
        source_row_id="P0C.R9.rec.win3.NEW_scene_drive (numbers.json)", status="final")
    for k, lab in (("normal", "Normal level"), ("win3", "win3 level"), ("gt_history", "GT-history level")):
        add(rows, claim_id=f"WINDOW.level.{k}", model=AV[0], domain=AV[1], evaluation_mode=AV[2],
            contrast_definition_and_sign=f"{lab} (numerator = Normal - win3 = {100*(lv['normal']-lv['win3']):.2f} pp; denominator = Normal - GT-history = {100*(lv['normal']-lv['gt_history']):.2f} pp)",
            cohort="dev A- stratum (R9)", estimand="A-", outcome=SEV, estimate=100 * lv[k], unit="% of units", ci_level=None,
            ci_method="NR (level, descriptive)", ci_use_and_limitations="descriptive", n_units=365, n_scenes=52, n_drives=13,
            weighting="scene-weighted", source_file="derived from outputs/review_followup4/A2_A3/a2_checks.json (ninety_percent_definition)",
            source_hash=sha(f"{O}/review_followup4/A2_A3/a2_checks.json"), source_row_id="scene_weighted_levels", status="derived")
    add(rows, claim_id="WINDOW.qualifiers", model=AV[0], domain=AV[1], evaluation_mode=AV[2],
        contrast_definition_and_sign=(f"GT positions per unit {a2['win3_mean_corrected_positions']:.2f}, differing from own token "
                                      f"{a2['win3_mean_effective_overwrites']:.2f}; steps after the last added GT position "
                                      f"{a2['win3_mean_post_release_steps']:.2f}; generated steps after t* {a2['mean_steps_generated_after_tstar']}"),
        cohort="dev A- stratum (R9)", estimand="A-", outcome="definition qualifiers", estimate=None, unit=None, ci_method="NR",
        n_units=365, n_scenes=52, n_drives=13, source_file="derived from outputs/review_followup4/A2_A3/a2_checks.json",
        source_hash=sha(f"{O}/review_followup4/A2_A3/a2_checks.json"), source_row_id="ninety_percent_definition", status="derived")
    # ---------------------------------------------------------------- SpatialVLA (A1 corrected, reused)
    cv_f = f"{O}/review_followup4/A1_A4/CI_VALIDITY_AUDIT.csv"
    cv = pd.read_csv(cv_f)
    cv = cv[cv.row_id.astype(str).str.startswith("P0B.P1D.")]
    for _, r in cv.iterrows():
        cfg, c, task, clu = r.row_id.split(".")[2:6]
        add(rows, claim_id=f"SVLA.{r.row_id}", model=SV[0], domain=SV[1], evaluation_mode=SV[2],
            contrast_definition_and_sign=("delta_R = success(R) - success(N): context-only perturbation" if c == "R-N" else
                                          "delta_C = success(C) - success(F): correction benefit (F - C = -delta_C)"),
            cohort=f"P1-D, config {cfg}", estimand="two-task mean (task-equal)" if task == "pooled" else f"task {task}",
            outcome="task success rate (closed loop), pp", estimate=f(r.estimate) * 100, unit="pp", ci_level=0.95,
            ci_low=f(r.new_ci95_lo) * 100, ci_high=f(r.new_ci95_hi) * 100,
            ci_method=str(r.new_ci_method) + f" ({clu} initial-state clusters)",
            ci_use_and_limitations="post-hoc corrected; approximate method (limited coverage simulation), move_near has 10 physical layouts",
            p_raw=f(r.new_p), p_adjusted=f(r.new_p_holm), test="cluster sign-flip at 0 (valid at 0)", null_assumptions="cluster totals sign-symmetric under H0",
            family=f"{r.family}; TOST +-10 pp Holm p_equiv = {f(r.new_p_equiv_m10_holm)}; +-5 pp Holm p_equiv = {f(r.new_p_equiv_m5_holm)}",
            alpha=0.05, n_units=int(r.n_units) if f(r.n_units) else None, n_scenes=None, n_drives=None,
            n_components=int(r.n_clusters) if f(r.n_clusters) else None, weighting="unit-weighted within task; task-equal mean",
            source_file=cv_f, source_row_id=r.row_id, status="final (review_followup4 A1)")
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/FINAL_PAPER_NUMBERS.csv", index=False)
    recs = json.loads(df.to_json(orient="records"))          # plain Python floats / null, no numpy reprs
    json.dump({"generated_by": __file__, "n_rows": len(recs), "rows": recs}, open(f"{OUT}/FINAL_PAPER_NUMBERS.json", "w"), indent=1)
    print(len(df), df.claim_id.str.split(".").str[0].value_counts().to_dict())


if __name__ == "__main__":
    main()
