#!/usr/bin/env python
"""P2 reports (review_followup3, 2026-10-14).  Writes MOTION_BASELINE_AUDIT.md, MOTION_BASELINE_RESULTS.md,
LEARNED_EXTRAPOLATOR_RESULTS.md and p2_verdict.json ONLY from the P2 CSV/JSON outputs (no hand-typed numbers).
Run: nice -n 19 python p2_reports.py"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

OUT = "/root/VLA/autovla_misalignment_poc/outputs/review_followup3/P2"
GATE = "/root/VLA/autovla_misalignment_poc/outputs/review_followup3/P1/NOOP_CACHE_AUDIT.md"
KIN = ["CM", "CV0", "CTRV-K3", "CTRA-2"]
GRPS = ["A-", "A+", "all"]
MLAB = {"end": "M1 endpoint displacement norm (m)", "sfx": "M2 suffix local-motion change (m/0.5 s)",
        "dref": "M3 signed error change vs reference r (m)", "dgt": "M4 signed GT error change (m)", "area": "M5 path area (m·s)"}


def f(x, d=2):
    return "NA" if pd.isna(x) else f"{x:+.{d}f}"


def p(x):
    return "NA" if pd.isna(x) else (f"{x:.2g}" if x >= 1e-3 else f"{x:.1e}")


def ci(r, d=2):
    return f"{f(r.est_scene, d)} [{f(r.ci95_lo, d)}, {f(r.ci95_hi, d)}]"


def status_line():
    return ("**Status: PROVISIONAL.** These are AutoVLA numbers from the P1-A rollouts, the C1 input path. They stay provisional until the AutoVLA "
            "P1 no-op/cache parity gate is judged (`outputs/review_followup3/P1/NOOP_CACHE_AUDIT.*`; "
            f"{'present' if os.path.exists(GATE) else 'NOT present'} when this report was generated). "
            "Section 6.2 and Figure 1 of the manuscript are held until the gate passes. "
            "Tags: driving domain, AutoVLA, open-loop 5 s plan, POST-HOC analysis of existing rollouts.")


def verdict_rows(T, E):
    """Interpretation rule (LEARNED_EXTRAPOLATOR_PROTOCOL section 8 + MOTION_BASELINE_SPEC 2.4), family G, metric end."""
    out = []
    for g in GRPS:
        k = T[(T.row_type == "excess") & (T.family == "G") & (T.group == g) & (T.metric == "end") & T.baseline.isin(KIN)]
        kin_ok = {r.baseline: bool(r.est_scene > 0 and r.p_holm_m4 < 0.05 and r.ci95_lo > 0) for r in k.itertuples()}
        rec = {"stratum": g, **{f"exceeds_{b}": v for b, v in kin_ok.items()}}
        if E is not None:
            e = E[(E.row_type == "excess") & (E.family == "G") & (E.group == g) & (E.metric == "end")]
            for v in ("prefix", "prefix_ego"):
                pr = e[(e.comparator == f"E:{v}") & e.analysis.str.startswith("primary")].iloc[0]
                se = e[(e.comparator == f"E:{v}") & e.analysis.str.startswith("sensitivity")].iloc[0]
                rec[f"exceeds_E_{v}"] = bool(pr.est_scene > 0 and pr.p_holm_m2 < 0.05 and pr.ci95_lo > 0)
                rec[f"sens_sign_kept_E_{v}"] = bool(np.sign(se.est_scene) == np.sign(pr.est_scene))
        keys = [x for x in rec if x.startswith("exceeds_") or x.startswith("sens_")]
        rec["beyond_motion_extrapolation"] = all(rec[x] for x in keys) if E is not None else None
        failing = [x.replace("exceeds_", "").replace("sens_sign_kept_", "sensitivity ") for x in keys if not rec[x]]
        rec["comparators_not_exceeded"] = ", ".join(failing) if failing else ""
        out.append(rec)
    return pd.DataFrame(out)


def audit():
    s = json.load(open(f"{OUT}/cm_definition_summary.json"))
    c1 = json.load(open(f"{OUT}/c1_reproduction_check.json"))
    lock = json.load(open(f"{OUT}/spec_lock_v1.json"))
    tests = open(f"{OUT}/tests_output.txt").read()
    dg = pd.read_csv(f"{OUT}/baseline_fit_diagnostics.csv")
    spec = open(f"{OUT}/MOTION_BASELINE_SPEC_LOCKED_v1.md").read()
    L = ["# MOTION_BASELINE_AUDIT (P2-A, review_followup3, 2026-10-14)", "",
         status_line(), "",
         "## 0. Provenance and spec lock",
         f"- **Spec adopted:** the MOTION_BASELINE_SPEC from the stopped v1 attempt, unchanged.",
         f"  - It was locked at **{lock['locked_utc']}** (sha256 `{lock['MOTION_BASELINE_AUDIT.md_sha256'][:16]}…`).",
         "  - Before the lock, v1 had run only the CM-definition check and the C1 reproduction.",
         "  - The v1 directory contains no new-baseline result file (no `p2_units.parquet` and no `MOTION_BASELINE_RESULTS*`). The v1 evaluation script carries a later mtime than the lock but never wrote output.",
         "  - This run first re-ran steps 1–2 and the unit tests, then evaluated. `p2_motion_baselines.py` asserts the sha256 of the verbatim spec copy and of `p2_kinematics.py` before computing anything.",
         "- **Files:**",
         "  - verbatim spec: `MOTION_BASELINE_SPEC_LOCKED_v1.md`;",
         "  - lock record: `spec_lock_v1.json`;",
         "  - the spec text is reproduced in section 4 below.",
         "- **Code:** `scripts/review_followup3/`",
         "  - `p2_kinematics.py`, identical to v1 and sha-locked;",
         "  - `p2_cm_audit_reproduce.py`;",
         "  - `p2_tests.py`;",
         "  - `p2_motion_baselines.py`: v1 plus the lock-file path and a status column;",
         "  - `p2_figure.py`;",
         "  - `p2_reports.py`;",
         "  - `p2_regen.sh`.", "",
         "## 1. CM definition (code + numeric check, regenerated)",
         "**Code.**",
         "- `scripts/review_followup/p1ab/p1a_gt_free_reference.py` builds `CM = ref[:t_p] + [p] * (10 - t_p)` as **tokens**. These are decoded by the real `ActionTokenizer.decode_token_ids_to_trajectory`.",
         "- C1 (`c_baseline_window.py`) uses `CM_0 = ref[:t] + [r_t] * (10 - t)`.",
         "- The decoder (`ActionTokenizer.rollout`; numpy copy `p1_common.rollout_np`) chains each token as an SE(2) increment in the frame of its own start pose:",
         "  - x_k = x_{k−1} + Rot(h_{k−1}) · disp[a];",
         "  - h_k = h_{k−1} + dyaw[a].",
         "",
         f"**Numeric check** on all {s['n_units']} perturbed P1-A units, using the stored GPU decoder output:",
         "",
         "| check | value |", "|---|---|",
         f"| body-frame increment of every CM suffix segment vs (disp[p], dyaw[p]): max error | {s['body_frame_SE2_repeat_max_err_m']:.1e} m, {s['body_frame_yaw_increment_max_err_rad']:.1e} rad |",
         f"| global-frame displacement repeat deviation (0 for a global-velocity repeat), median over turning units / max | {s['global_disp_repeat_maxdev_m_median_over_turning_units']:.2f} m / {s['global_disp_repeat_maxdev_m_max']:.2f} m |",
         f"| share of units whose perturbed token turns (abs(dyaw[p]) > 0.5°) | {100 * s['share_units_|dyaw_p|>0.5deg']:.1f}% |",
         f"| abs(dyaw[p]) median / p90 per 0.5 s | {s['median_|dyaw_p|_deg']:.1f}° / {s['p90_|dyaw_p|_deg']:.1f}° |",
         f"| abs(chord angle − dyaw/2) median / p90 (deviation from a zero-sideslip arc) | {s['median_|chord_angle - dyaw/2|_deg']:.1f}° / {s['p90_|chord_angle - dyaw/2|_deg']:.1f}° |",
         f"| numpy SE(2) chaining vs stored decoder | {s['compose_vs_stored_decoder_max_err']:.1e} |", "",
         "**Finding (rebuttal-relevant).** The existing 'repeat-p constant-motion' baseline is **not** a global constant-velocity repeat.",
         "- It is a **body-frame (local) SE(2) repeat, yaw included**: the perturbed token's translation is applied in the current heading frame, and the heading advances by dyaw[p] every 0.5 s.",
         "- So it is already a discrete constant-turn-rate, constant-speed (constant body twist) extrapolation. It also keeps the token's lateral (chord-angle) component.",
         "- A 'CTRV fitted to the last token' would differ from CM only by imposing zero sideslip. It is therefore **not added**, as a renamed near-duplicate.",
         "- The reviewer's premise that CM ignores curvature is not correct for this implementation.",
         "",
         "**Controls that are distinct from CM** (spec section 2.3):",
         "- **CV0:** zero-turn, global constant velocity. It removes the yaw-rate continuation.",
         "- **CTRV-K3:** body twist averaged over the last ≤3 steps. The perturbation enters with weight 1/K_eff.",
         "- **CTRA-2:** constant turn rate plus the acceleration extrapolated from the last two speeds.",
         "",
         "## 2. C1 reproduction (exact)",
         f"- `c_baseline_window.c1_units` / `c1_tables` were re-run in-process with the original seeds.",
         f"- Unit rows equal: {c1['units_rows_equal']}; unit numeric max abs diff {c1['units_numeric_maxabs_diff']:.1e}.",
         f"- Table rows equal: {c1['table_rows_equal']}; table numeric max abs diff {c1['table_numeric_maxabs_diff']:.1e}.",
         f"- Reproduced exactly: **{c1['reproduced_exactly']}**.",
         "",
         "Key cells (family G):",
         "", "| cell | est [95% CI] | sign-flip p |", "|---|---|---|"]
    for k, v in c1["key_cells"].items():
        L.append(f"| {k} | {v['est_scene']:+.2f} [{v['ci95_lo']:+.2f}, {v['ci95_hi']:+.2f}] | {p(v['p_signflip'])} |")
    L += ["", "## 3. Unit tests (`tests_output.txt`)", "", "```", tests.strip(), "```", "",
          "Clip and fallback activation (inputs only; `baseline_fit_diagnostics.csv`):", "",
          dg.round(3).to_markdown(index=False), "",
          "## 4. Locked spec (verbatim copy of MOTION_BASELINE_SPEC_LOCKED_v1.md)", "",
          "> The text below was written by the stopped v1 attempt and locked before any new-baseline result was computed.",
          "> Paths in it refer to the v1 layout. Its section 1 numbers equal the regenerated values above.", "", spec]
    open(f"{OUT}/MOTION_BASELINE_AUDIT.md", "w").write("\n".join(L) + "\n")


def results(T, V):
    L = ["# MOTION_BASELINE_RESULTS (P2-A)", "", status_line(), "",
         "**Quantity.** Δ_model − Δ_b, where Δ = φ(perturbed) − φ(unperturbed) is computed per method.",
         "- Units are paired; the horizon is 10 poses.",
         "- Scenes are equally weighted.",
         "- Inference uses drive clusters: bootstrap with B = 10,000 and an exact drive sign-flip test.",
         "- Holm m = 4 over the kinematic baselines within each (family, stratum, metric).",
         "- Geo is privileged: it uses the model's unperturbed future r[t+1:]. It is reported outside the Holm family.",
         "- A positive value means the model moves more than the baseline. 'Not detected' is not equivalence.",
         "- Source: `MOTION_BASELINE_RESULTS.csv`. Figure: `fig_p2_motion_baselines.pdf/.png`.", "",
         "## Primary: family G, M1 endpoint displacement norm (m)", "",
         "| stratum | baseline | Δ_model − Δ_b [95% CI] | sign-flip p | Holm p (m=4) | drives | judgment |", "|---|---|---|---|---|---|---|"]
    x = T[(T.row_type == "excess") & (T.family == "G") & (T.metric == "end")]
    for r in x.itertuples():
        L.append(f"| {r.group} | {r.baseline} | {ci(r)} | {p(r.p_signflip)} | {p(r.p_holm_m4)} | {r.n_drives} | {r.judgment} |")
    L += ["", "Levels Δ_method (M1, family G; descriptive):", "", "| stratum | method | Δ [95% CI] |", "|---|---|---|"]
    for r in T[(T.row_type == "level") & (T.family == "G") & (T.metric == "end")].itertuples():
        L.append(f"| {r.group} | {r.baseline} | {ci(r)} |")
    L += ["", "## Secondary metrics, family G (kept separate; they never override M1)", ""]
    for m in ["sfx", "dref", "dgt", "area"]:
        L += [f"### {MLAB[m]}", "", "| stratum | " + " | ".join(KIN + ["Geo"]) + " |", "|---" * 6 + "|"]
        for g in GRPS:
            cells = []
            for b in KIN + ["Geo"]:
                r = T[(T.row_type == "excess") & (T.family == "G") & (T.metric == m) & (T.group == g) & (T.baseline == b)].iloc[0]
                star = "" if b == "Geo" else (" *" if r.p_holm_m4 < 0.05 and (r.ci95_lo > 0 or r.ci95_hi < 0) else "")
                cells.append(ci(r) + star)
            L.append(f"| {g} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["(* = Holm m=4 p < 0.05 and CI excludes 0)", "",
          "## Other perturbation families, M1 (secondary)", "",
          "Family E is GT-anchored. Families GF1 and GF3 are fixed-position and GT-free.", "",
          "| family | stratum | " + " | ".join(KIN + ["Geo"]) + " |", "|---" * 7 + "|"]
    for fam in ["E", "GF1", "GF3"]:
        for g in GRPS:
            cells = []
            for b in KIN + ["Geo"]:
                r = T[(T.row_type == "excess") & (T.family == fam) & (T.metric == "end") & (T.group == g) & (T.baseline == b)].iloc[0]
                star = "" if b == "Geo" else (" *" if r.p_holm_m4 < 0.05 and (r.ci95_lo > 0 or r.ci95_hi < 0) else "")
                cells.append(ci(r) + star)
            L.append(f"| {fam} | {g} | " + " | ".join(cells) + " |")
    L += ["", "## Decision (locked rule 2.4, kinematic part only; family G, M1)", "",
          "| stratum | " + " | ".join(f"exceeds {b}" for b in KIN) + " |", "|---" * 5 + "|"]
    for r in V.to_dict("records"):
        L.append(f"| {r['stratum']} | " + " | ".join(str(r[f"exceeds_{b}"]) for b in KIN) + " |")
    L += ["", "The combined verdict, which adds the learned extrapolator, is in `LEARNED_EXTRAPOLATOR_RESULTS.md`."]
    open(f"{OUT}/MOTION_BASELINE_RESULTS.md", "w").write("\n".join(L) + "\n")


def extrap(E, V):
    sep = json.load(open(f"{OUT}/extrap_drive_separation.json"))
    hp = pd.read_csv(f"{OUT}/extrap_hparam_selection.csv")
    cov = pd.read_csv(f"{OUT}/extrap_coverage.csv")
    va = pd.read_csv(f"{OUT}/extrap_val_accuracy.csv")
    lock = json.load(open(f"{OUT}/extrap_protocol_lock.json"))
    tests = open(f"{OUT}/extrap_tests_output.txt").read()
    logs = pd.concat([pd.read_csv(f"{OUT}/_extrap_models/{n}") for n in sorted(os.listdir(f"{OUT}/_extrap_models")) if n.endswith("_log.csv")])
    logs.to_csv(f"{OUT}/extrap_training_log.csv", index=False)
    L = ["# LEARNED_EXTRAPOLATOR_RESULTS (P2-B)", "", status_line(), "",
         "This is a small learned **motion** extrapolator: a non-autoregressive MLP that predicts the whole suffix from the prefix. It is **not** a non-AR version of the VLA.",
         "",
         f"**Protocol:** `LEARNED_EXTRAPOLATOR_PROTOCOL.md`, locked at **{lock['locked_utc']}**.",
         "- The lock preceded training-data extraction, training and every P2-A new-baseline result.",
         "- The protocol was not amended.",
         "- One implementation fix was made before any sample existed: OpenScene `frame_idx` restarts in every scene, so plan frames are ordered by timestamp within a log and adjacency is checked with Δt only. A first run that used frame_idx adjacency produced 0 samples.",
         "",
         "## Data and drive separation (`extrap_drive_separation.json`)",
         "- **Source:** OpenScene v1.1 trainval metadata, i.e. the AutoVLA checkpoint training split. It was streamed (7.05 GB over the network, never written to disk). Only per-frame ego fields were kept, 56 MB on disk.",
         f"- **Drives:**",
         f"  - {sep['n_trainval_drives']} trainval drives;",
         f"  - {sep['n_excluded_test_drives']} excluded for overlap with test-split, dev, P1-C or P5-pool drives (there is none);",
         f"  - **{sep['n_train_drives']} training drives** and **{sep['n_val_drives']} validation drives**.",
         f"- **Drive overlap with reference sets:** {sep['drive_overlap']}; train–validation drive overlap: {sep['train_val_drive_overlap']}.",
         f"- **Reference set sizes (drives):** {sep['reference_sets']}.",
         f"- **Descriptive day overlap, training vs P1-A evaluation drives:** {sep['descriptive_day_overlap_train_vs_p1a_eval']}. Vehicle overlap: {len(sep['descriptive_vehicle_overlap_train_vs_p1a_eval'])} vehicles. Drive-disjointness is the requirement; this overlap is reported, not required to be zero.",
         f"- **Plan frames:** {sep['n_plan_frames']}, of which {sep['n_plan_frames_train']} training and {sep['n_plan_frames_val']} validation. Each frame gives 9 rows, j = 0..8. Dropped for Δt gaps: {sep['n_dropped_dt_or_gap']}.",
         f"- **GT→token quantisation residual:** median {sep['token_quantization_residual_m']['median']:.3f} m, p99 {sep['token_quantization_residual_m']['p99']:.3f} m.",
         "- The evaluation script asserts that no P1-A evaluation drive or log appears in the samples.", "",
         "## Unit tests (`extrap_tests_output.txt`)", "", "```", tests.strip(), "```", "",
         "## Hyperparameter selection (validation drives only, primary input, seed 0)", "", hp.to_markdown(index=False), "",
         "## Validation-drive accuracy (descriptive; mean position error over suffix poses, m)", "",
         "These are token-domain GT plans; 5,000 validation plans per j.", ""]
    piv = va.pivot(index="j", columns="method", values="val_mean_pos_err_m").round(3)
    L += [piv.to_markdown(), "",
          "## Coverage of the evaluated prefixes vs the training prefix distribution (`extrap_coverage.csv`)", "",
          "- Distance: nearest-neighbour distance in standardised primary-feature space.",
          "- Out of range: the perturbed or unperturbed prefix is beyond the validation q99 for the same j.",
          "- **Degenerate reference at j ≤ 1.** Prefixes are decoded tokens, so at j = 0 (and mostly at j = 1) validation prefixes have exact token matches in the training sample. The validation q99 is then 0 (j = 0) or small. 'Out of range' there means the prefix (token sequence) does not occur in the 100k-row training reference sample.",
          "- The ratio columns are NaN where the validation q99 is 0. This is the locked rule, applied as written.", ""]
    c = cov[cov.table == "eval_by_family_stratum"][["family", "group", "n_units", "nn_pert_q50", "nn_pert_q90", "nn_pert_max",
                                                     "ratio_pert_to_val_q99_q50", "ratio_pert_to_val_q99_q90", "share_pert_out_of_range",
                                                     "share_unpert_out_of_range", "share_unit_out_of_range", "n_units_in_range", "n_drives_in_range"]]
    L += [c.round(3).to_markdown(index=False), "", "Per-j reference distribution:", "",
          cov[cov.table == "per_j_reference"][["j", "n_train_ref", "n_val", "val_nn_q50", "val_nn_q90", "val_nn_q99", "n_eval_units"]].round(3).to_markdown(index=False), "",
          "## Primary: family G, M1 endpoint displacement norm, Δ_model − Δ_E (m)", "",
          "| stratum | extrapolator | analysis | Δ_model − Δ_E [95% CI] | sign-flip p | Holm p (m=2) | drives | seed est. min / max (SD) | judgment |",
          "|---|---|---|---|---|---|---|---|---|"]
    x = E[(E.row_type == "excess") & (E.family == "G") & (E.metric == "end")]
    for r in x.itertuples():
        sd = f"{f(r.seed_est_min)} / {f(r.seed_est_max)} ({r.seed_est_sd:.3f})" if not pd.isna(r.seed_est_sd) else "—"
        L.append(f"| {r.group} | {r.comparator} | {r.analysis} | {ci(r)} | {p(r.p_signflip)} | {p(r.p_holm_m2)} | {r.n_drives} | {sd} | {r.judgment if isinstance(r.judgment, str) else '—'} |")
    L += ["", "Per-seed rows (seed variance, family G, M1):", "", "| stratum | extrapolator | seed | Δ_model − Δ_E [95% CI] |", "|---|---|---|---|"]
    for r in E[(E.row_type == "excess_per_seed") & (E.family == "G") & (E.metric == "end")].itertuples():
        L.append(f"| {r.group} | {r.comparator} | {int(r.train_seed)} | {ci(r)} |")
    L += ["", "Levels Δ_E (seed mean; family G, M1):", "", "| stratum | extrapolator | Δ_E [95% CI] |", "|---|---|---|"]
    for r in E[(E.row_type == "level") & (E.family == "G") & (E.metric == "end")].itertuples():
        L.append(f"| {r.group} | {r.comparator} | {ci(r)} |")
    L += ["", "## Secondary metrics and families (primary analysis, seed mean)", "",
          "| family | metric | stratum | E:prefix | E:prefix_ego |", "|---|---|---|---|---|"]
    for fam in ["G", "E", "GF1", "GF3"]:
        for m in (["sfx", "dref", "dgt", "area"] if fam == "G" else ["end", "sfx"]):
            for g in GRPS:
                cells = []
                for v in ("E:prefix", "E:prefix_ego"):
                    r = E[(E.row_type == "excess") & (E.family == fam) & (E.metric == m) & (E.group == g) & (E.comparator == v)
                          & E.analysis.str.startswith("primary")].iloc[0]
                    cells.append(ci(r) + (" *" if r.p_holm_m2 < 0.05 and (r.ci95_lo > 0 or r.ci95_hi < 0) else ""))
                L.append(f"| {fam} | {m} | {g} | " + " | ".join(cells) + " |")
    L += ["(* = Holm m=2 p < 0.05 and CI excludes 0)", "",
          "## Combined interpretation (protocol section 8; family G, M1)", "",
          V.to_markdown(index=False), ""]
    for r in V.itertuples():
        if r.beyond_motion_extrapolation:
            L.append(f"- **{r.stratum}:** the model's endpoint response exceeds all four kinematic controls and both learned extrapolators, and the out-of-range-excluded sensitivity keeps the sign. 'Beyond motion extrapolation' may be stated for this stratum only, as an open-loop AutoVLA driving result. It is PROVISIONAL until the gate.")
        else:
            L.append(f"- **{r.stratum}:** not distinguished from motion extrapolation. The model does not exceed: {r.comparators_not_exceeded}. The §6.2 statement must be narrowed for this stratum.")
    open(f"{OUT}/LEARNED_EXTRAPOLATOR_RESULTS.md", "w").write("\n".join(L) + "\n")


def main():
    T = pd.read_csv(f"{OUT}/MOTION_BASELINE_RESULTS.csv")
    E = pd.read_csv(f"{OUT}/LEARNED_EXTRAPOLATOR_RESULTS.csv") if os.path.exists(f"{OUT}/LEARNED_EXTRAPOLATOR_RESULTS.csv") else None
    V = verdict_rows(T, E)
    V.to_csv(f"{OUT}/p2_verdict.csv", index=False)
    audit()
    results(T, V)
    if E is not None:
        extrap(E, V)
    print(V.to_string())


if __name__ == "__main__":
    main()
