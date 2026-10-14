#!/usr/bin/env python
"""
P0 merger (review_followup3, 2026-10-14): DATA_MANIFEST.csv and numbers_p0.json, built ONLY from the P0 output tables
(no number is typed by hand).  POST-HOC bookkeeping, CPU only.

DATA_MANIFEST.csv  one row per independent-unit record with the ID hierarchy kept separate
                   (log / drive / day / vehicle / day-vehicle / scene / perturbation / initial physical state /
                   instruction / RNG seed / task), inclusion flag, missing reason, denominator note and status:
                   AutoVLA dev natural scenes (2,748) and P1-C stage-1 scenes (5,671)  [p0_a]
                   SpatialVLA units (exp 34 / exp 35 / P1-D)                             [p0_b]
                   OpenVLA LIBERO-Spatial initial states (100)                           [p0_e]
                   drive map (44 drives; set membership, training overlap)               [p0_d]
numbers_p0.json    {row_id: {value, ci95, p_raw, p_adj, family, n, status, source_path, ...}} for every P0 table row.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import pandas as pd

POC = "/root/VLA/autovla_misalignment_poc"
OUT = f"{POC}/outputs/review_followup3/P0"
PROV = "PROVISIONAL (AutoVLA no-op parity, P1 Stage 0, not judged)"


def clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def rec(**k):
    return {a: clean(b) for a, b in k.items() if not (isinstance(b, float) and not math.isfinite(b))}


def manifest():
    A = pd.read_csv(f"{OUT}/_frag_manifest_A.csv", low_memory=False)
    A["status"] = PROV
    Bm = pd.read_csv(f"{OUT}/_frag_manifest_B.csv")
    Bm["status"] = np.where(Bm.experiment == "exp35", "SpatialVLA; exp 35 restore NOT verified",
                            np.where(Bm.experiment == "P1D", "SpatialVLA; P1-D restore verified",
                                     "SpatialVLA; exp 34 restore not verified (t=0 frame check only)"))
    D = pd.read_csv(f"{OUT}/_frag_manifest_D.csv")
    D["status"] = PROV + " (drive map)"
    E = pd.read_csv(f"{OUT}/OPENVLA_episode_records.csv")
    U = pd.read_csv(f"{OUT}/OPENVLA_phaseC_feedback_units.csv", usecols=["task", "init", "frame"])
    nfr = U.drop_duplicates("frame").groupby(["task", "init"]).size()
    er = []
    for (t, i), g in E.groupby(["task", "init"]):
        er.append({"dataset": "OpenVLA LIBERO-Spatial (closed loop + offline frames)", "experiment": "exp 28 (cross_vla_replication)",
                   "model": "openvla-7b-finetuned-libero-spatial (HF 962318ce)", "task": f"t{t}: {g.task_name.iloc[0]}",
                   "unit_id": f"t{t}_i{i}", "scene": f"t{t}_i{i}", "perturbation": f"x token, |delta| 8/24, sign {'+' if i % 2 == 0 else '-'} (even init +), control steps 10-19",
                   "initial_physical_state": g.initial_state_id.iloc[0], "instruction": g.task_name.iloc[0].replace("_", " "),
                   "rng_seed": "greedy decode; env.seed(0)", "condition_set": ";".join(sorted(g.cond)), "n_rows": len(g),
                   "included": True, "missing_reason": "",
                   "denominator_note": f"1 episode per condition (8); phase-C frames {int(nfr.get((t, i), 0))}; "
                                       f"natural success {int(g[g.cond == 'natural'].success.iloc[0])}",
                   "status": "OpenVLA post-hoc (not subject to AutoVLA parity; OpenVLA path not parity-verified)"})
    M = pd.concat([A, Bm, pd.DataFrame(er), D], ignore_index=True)
    M.insert(0, "manifest_row", range(len(M)))
    M.to_csv(f"{OUT}/DATA_MANIFEST.csv", index=False)
    return M


def numbers():
    N = {}
    src = lambda f: f"outputs/review_followup3/P0/{f}"  # noqa: E731
    # ---- P0-A
    for r in pd.read_csv(f"{OUT}/POPULATION_EFFECTS.csv").itertuples():
        N[r.row_id] = rec(item="P0-A effect", contrast=r.contrast_name, estimand=r.estimand, value=r.theta, unit="proportion (x100 = pp)",
                          ci95_boot=[clean(r.boot_ci95_lo), clean(r.boot_ci95_hi)], ci95_signflip=[clean(r.sf_ci95_lo), clean(r.sf_ci95_hi)],
                          p_raw=r.p_signflip_raw, p_holm=r.p_holm, family=r.family_id, n_scenes=r.n_scenes, n_units=r.n_units,
                          n_drives=r.n_clusters, n_informative_drives=r.n_informative_clusters, events_a=r.events_a, n_a=r.n_a,
                          events_b=r.events_b, n_b=r.n_b, status=r.status, source_path=src("POPULATION_EFFECTS.csv"))
    for r in pd.read_csv(f"{OUT}/POPULATION_prevalence.csv").itertuples():
        k = f"P0A.PREV.{'DEV' if r.cohort.startswith('dev') else 'P1C'}.{r.outcome}.{r.denominator.split()[0].replace(',', '')}.{r.cluster_level}"
        N[k] = rec(item="P0-A prevalence (evaluation-set frequency, NOT deployment)", value=r.prevalence, events=r.events, n=r.n_scenes,
                   denominator=r.denominator, cluster=r.cluster_level, n_clusters=r.n_clusters, ci95_boot=[clean(r.boot_ci95_lo), clean(r.boot_ci95_hi)],
                   status=r.status, source_path=src("POPULATION_prevalence.csv"))
    for r in pd.read_csv(f"{OUT}/POPULATION_decomposition.csv").itertuples():
        N[f"P0A.DECOMP.{r.contrast}"] = rec(item="P1-C P = F + S decomposition", value=r.theta_P, theta_F=r.theta_F, theta_S=r.theta_S_hajek,
                                            share_F=r.share_F, F_contribution=r.F_contribution_to_theta_P, S_contribution=r.S_contribution_to_theta_P,
                                            status=r.status, source_path=src("POPULATION_decomposition.csv"))
    for r in pd.read_csv(f"{OUT}/POPULATION_selection_flow.csv").itertuples():
        k = f"P0A.FLOW.{'DEV' if r.cohort.startswith('dev') else 'P1C'}.{r.step.split()[0]}"
        N[k] = rec(item="selection flow", step=r.step, n_scenes=r.n_scenes, n_logs=r.n_logs, n_drives=r.n_drives, n_days=r.n_days,
                   n_vehicles=r.n_vehicles, n_day_vehicles=r.n_day_vehicles, status=PROV, source_path=src("POPULATION_selection_flow.csv"))
    # ---- P0-B
    for f in ("SPATIAL_TASK_EFFECTS.csv", "SPATIAL_TASK_EFFECTS_equal_task_mean.csv"):
        for r in pd.read_csv(f"{OUT}/{f}").itertuples():
            N[r.row_id] = rec(item="P0-B direct effect", value_pp=r.estimate_pp, ci95_signflip_inv_pp=[clean(r.ci95_lo_pp), clean(r.ci95_hi_pp)],
                              ci95_boot_pp=[clean(r.boot_ci95_lo_pp), clean(r.boot_ci95_hi_pp)], p_raw=r.p_diff_signflip, p_holm=r.p_diff_holm,
                              family=r.family_id, family_m=r.family_m, p_equiv_m10=r.p_equiv_m10, p_equiv_m10_holm=r.p_equiv_m10_holm,
                              p_equiv_m5=r.p_equiv_m5, p_equiv_m5_holm=r.p_equiv_m5_holm, n_units=r.n_units, n_clusters=r.n_clusters_used,
                              succ=[clean(r.succ_N), clean(r.succ_F), clean(r.succ_C), clean(r.succ_R)], att=clean(r.att_N),
                              status=r.status, restore=r.restore_verification, source_path=src(f))
    for r in pd.read_csv(f"{OUT}/SPATIAL_TASK_EFFECTS_heterogeneity.csv").itertuples():
        N[r.row_id] = rec(item="P0-B task x intervention heterogeneity", value_pp=r.heterogeneity_coke_minus_move_near_pp,
                          ci95_boot_pp=[clean(r.het_boot_ci95_lo_pp), clean(r.het_boot_ci95_hi_pp)], p_welch=r.het_welch_p,
                          coke_pp=r.coke_pp, move_near_pp=r.move_near_pp, equal_task_mean_pp=r.equal_task_mean_pp,
                          cancellation_share=r.cancellation_share, status="POST-HOC (SpatialVLA); unadjusted",
                          source_path=src("SPATIAL_TASK_EFFECTS_heterogeneity.csv"))
    for r in pd.read_csv(f"{OUT}/SPATIAL_TASK_EFFECTS_weighting.csv").itertuples():
        N[r.row_id] = rec(item="P0-B weighting sensitivity", weighting=r.weighting, value_pp=r.estimate_pp,
                          ci95_pp=[clean(r.ci95_lo_pp), clean(r.ci95_hi_pp)], p_raw=r.p_signflip, p_equiv_m10=r.p_equiv_m10,
                          p_equiv_m5=r.p_equiv_m5, status=r.status, source_path=src("SPATIAL_TASK_EFFECTS_weighting.csv"))
    # ---- P0-C
    V = pd.read_csv(f"{OUT}/RECON_plus437_versions.csv")
    for r in V.itertuples():
        N[f"P0C.plus437.{r.version}"] = rec(item="+4.37 m version", label=r.label, metric=r.metric, value_m=r.estimate,
                                            ci95=[clean(r.ci95_lo), clean(r.ci95_hi)], p_raw=r.p_raw, p_holm=r.p_holm, holm_family=r.holm_family,
                                            cluster=r.cluster_level, bootstrap=r.bootstrap, status=PROV, source_path=r.source_file,
                                            source_code=r.source_code)
    v2 = V[V.version == "V2"].iloc[0]
    N["P0C.plus437.AUTHORITATIVE"] = rec(item="+4.37 m AUTHORITATIVE row", definition="predeclared P1A-2 estimand (OC - Geo, dev9_ref = ||X9 - r9||, "
                                         "family G, A-, scene-weighted) with drive-cluster inference as stored by review_followup2 C1",
                                         value_m=v2.estimate, ci95=[clean(v2.ci95_lo), clean(v2.ci95_hi)],
                                         ci95_signflip_inv=[clean(v2.sf_ci95_lo), clean(v2.sf_ci95_hi)], p_raw=v2.p_raw, p_holm=v2.p_holm,
                                         holm_family="predeclared P1A family m = 3", n_units=v2.n_units, n_scenes=v2.n_scenes, n_drives=v2.n_clusters,
                                         quote=(f"{v2.estimate:+.2f} m [{v2.ci95_lo:+.2f}, {v2.ci95_hi:+.2f}] (drive bootstrap); sign-flip inversion "
                                                f"[{v2.sf_ci95_lo:+.2f}, {v2.sf_ci95_hi:+.2f}]; p = {v2.p_raw:.2g} (floor 2/2^13), Holm {v2.p_holm:.2g}"),
                                         status=PROV, source_path=v2.source_file)
    for r in pd.read_csv(f"{OUT}/RECON_doc_number_check.csv").itertuples(index=True):
        if r.verdict != "OK":
            N[f"P0C.DOC.{r.Index:03d}"] = rec(item="document number check (non-OK)", check=r.check_id, doc=r.doc, doc_line=r.doc_line,
                                              quoted=r.quoted, source_value=r.source_full_precision, verdict=r.verdict,
                                              status="audit", source_path=src("RECON_doc_number_check.csv"))
    for r in pd.read_csv(f"{OUT}/WINDOW_WEIGHT_AUDIT.csv", low_memory=False).itertuples():
        d = r._asdict()
        val = d.get("fraction") if pd.notna(d.get("fraction")) else d.get("estimate")
        N[r.row_id] = rec(item="P0-C window/weight audit", table=d.get("table"), condition=d.get("condition"), estimator=d.get("estimator"),
                          value=val, ci95=[clean(d.get("ci95_lo")), clean(d.get("ci95_hi"))], n_units=d.get("n_units"), n_scenes=d.get("n_scenes"),
                          n_logs=d.get("n_logs"), n_drives=d.get("n_drives"), n_day_vehicles=d.get("n_day_vehicles"),
                          post_release_mean=d.get("post_release_free_steps_mean"), status=d.get("status"), source_path=src("WINDOW_WEIGHT_AUDIT.csv"))
    # ---- P0-D
    for r in pd.read_csv(f"{OUT}/DEPENDENCE_SENSITIVITY.csv").itertuples():
        N[r.row_id] = rec(item="P0-D dependence sensitivity", contrast=r.contrast_id, axis=r.axis, G=str(r.G), value=r.theta,
                          ci95_boot=[clean(r.boot_ci95_lo), clean(r.boot_ci95_hi)], ci95_signflip=[clean(r.sf_ci95_lo), clean(r.sf_ci95_hi)],
                          p_raw=r.p_signflip, p_min_attainable=r.p_min_attainable, p_holm=r.holm_p, status=r.status,
                          source_path=src("DEPENDENCE_SENSITIVITY.csv"))
    for i, r in enumerate(pd.read_csv(f"{OUT}/DISJOINT_LOG_INVENTORY_summary.csv").itertuples()):
        N[f"P0D.DISJ.{i:02d}"] = rec(item="disjoint-log inventory", subset=r.subset, n_logs=r.n_logs, n_drives=r.n_drives, n_days=r.n_days,
                                     n_vehicles=r.n_vehicles, n_day_vehicles=r.n_day_vehicles, navtest_scenes=r.navtest_scenes,
                                     exp_eligible_E=r.exp_eligible_E, exp_F=r.exp_F, status="inventory (data availability; not an effect); exp_eligible_E / exp_F use AutoVLA P1-C stage-1 rates -> " + PROV,
                                     source_path=src("DISJOINT_LOG_INVENTORY_summary.csv"))
    # ---- P0-E
    for r in pd.read_csv(f"{OUT}/OPENVLA_REANALYSIS.csv").itertuples():
        N[r.row_id] = rec(item="P0-E OpenVLA success contrast", contrast=r.contrast, value=r.estimate,
                          ci95_state_boot=[r.state_boot_ci95_lo, r.state_boot_ci95_hi], ci95_state_signflip_inv=[r.state_sf_inv_ci95_lo, r.state_sf_inv_ci95_hi],
                          ci95_task_strat_boot=[r.task_strat_boot_ci95_lo, r.task_strat_boot_ci95_hi], p_raw=r.p_signflip_state_exact,
                          p_holm=r.p_holm_state, family=r.family, n_states=r.n_initial_states, n_informative=r.n_informative_states,
                          old_value=r.OLD_estimate, old_ci95_task_boot=[r.OLD_task_boot_ci95_lo, r.OLD_task_boot_ci95_hi], old_p_mcnemar=r.OLD_p_mcnemar,
                          status=r.status, source_path=src("OPENVLA_REANALYSIS.csv"))
    for _, r in pd.read_csv(f"{OUT}/OPENVLA_REANALYSIS_per_task.csv").iterrows():
        N[r.row_id] = rec(item="P0-E per-task (10 states; low precision)", contrast=r.contrast, task=r.task, value=r.estimate,
                          p_raw=r["p_signflip_exact_2^10"], ci95_signflip_inv=[clean(r.sf_inv_ci95_lo), clean(r.sf_inv_ci95_hi)],
                          n_states=r.n_states, status=r.status, source_path=src("OPENVLA_REANALYSIS_per_task.csv"))
    for r in pd.read_csv(f"{OUT}/OPENVLA_REANALYSIS_phaseC.csv").itertuples():
        N[r.row_id] = rec(item="P0-E within-step amplification", value=r.amplified_rate, n_units=r.n_units, n_states=r.n_initial_states,
                          ci95_state_boot=[r.state_boot_ci95_lo, r.state_boot_ci95_hi], ci95_task_boot_old=[clean(r.OLD_task_boot_ci95_lo), clean(r.OLD_task_boot_ci95_hi)],
                          equal_state_weight=r.equal_state_weight_rate, phys_amp_translation=r.phys_amp_translation_rate,
                          injected_phys=r.injected_phys_mean, status=r.status, source_path=src("OPENVLA_REANALYSIS_phaseC.csv"))
    for r in pd.read_csv(f"{OUT}/OPENVLA_REANALYSIS_phaseC_dims.csv").itertuples():
        N[r.row_id] = rec(item="P0-E per-dimension decomposition", share_ge_delta=r.share_dbin_ge_delta, share_changed=r.share_dbin_gt0,
                          mean_abs_dbin=r.mean_abs_dbin, mean_abs_phys=r.mean_abs_phys, physical_unit=r.physical_unit,
                          status=r.status, source_path=src("OPENVLA_REANALYSIS_phaseC_dims.csv"))
    meta = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_rows": len(N),
            "rule": "every value read from a P0 output table; AutoVLA-derived rows carry PROVISIONAL until the AutoVLA no-op parity "
                    "(P1 Stage 0) is judged; SpatialVLA rows: exp 35 restore not verified, P1-D verified; OpenVLA rows post-hoc",
            "generator": "scripts/review_followup3/p0_manifest_numbers.py"}
    json.dump({"_meta": meta, "rows": N}, open(f"{OUT}/numbers_p0.json", "w"), indent=1, default=clean)
    return N


if __name__ == "__main__":
    M = manifest()
    N = numbers()
    print("manifest rows", len(M), M.dataset.value_counts().to_dict())
    print("numbers rows", len(N))
    st = pd.Series([v.get("status", "") for v in N.values()]).str[:40].value_counts()
    print(st.to_string())
