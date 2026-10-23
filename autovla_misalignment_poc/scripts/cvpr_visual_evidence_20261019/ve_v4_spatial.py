#!/usr/bin/env python
"""V4 (post-hoc): SpatialVLA P1-D episodes -> episode table, per-step series, paired TCP deviation, success, cases.
No RGB was stored (only image hashes) -> TCP / success only. CPU only.
Run: nice -n 19 /root/miniforge3/envs/autovla/bin/python ve_v4_spatial.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ve_common as V  # noqa: E402

OUT = V.OUT
SRC = f"{V.P1D}/rollouts/main/episodes.jsonl"
ARMS = {"natural": "N", "feedback": "F", "corrected": "C", "reverse": "R", "feedbackD": "FD", "correctedD": "CD",
        "null": "null", "gen": "gen"}


def main():
    eps, ts, pos = [], [], {}
    for line in open(SRC):
        e = json.loads(line)
        arm = ARMS.get(e["mode"], e["mode"])
        ok = None if e["restore"] is None else bool(e["restore"].get("ok"))
        key = (e["task"], e["seed"], e["cfg"], arm)
        eps.append({"task": e["task"], "seed": e["seed"], "episode_id": e["episode_id"], "instruction": e["instr"],
                    "overlay": e["overlay"], "config": e["cfg"], "arm": arm, "cond": e["cond"],
                    "physical_state_hash": e["init"]["state_hash"], "initial_img_hash": e["init"]["img_hash"],
                    "physical_state_id": f"{e['task']}|{e['init']['state_hash']}", "restore_ok": ok,
                    "eligible": ok is not False, "success": bool(e["success"]), "episode_len": e["steps"],
                    "T0_branch": e["T0"], "inject_steps": e["inject"], "n_gen_calls": e["n_gen_calls"],
                    "tcp_available": True, "rgb_available": False,
                    "ever_grasped": bool(e["ever"].get("is_grasped", False))})
        P = np.array([r["pose"] for r in e["rec"]], float)
        pos[key] = {"pos": P[:, :3], "act": np.array([r["action"] for r in e["rec"]], float), "success": int(e["success"]),
                    "ok": ok, "state": e["init"]["state_hash"]}
        if arm in ("N", "F", "C", "R"):
            for r in e["rec"]:
                pr = r["preds"][0] if r["preds"] else [np.nan] * 3
                ts.append((e["task"], e["seed"], e["cfg"], arm, r["t"], *r["action"], *r["pose"], r["grip"], *pr,
                           r.get("nat1"), r.get("ctx") is not None, len(r["preds"]), r["state_hash"]))
    E = pd.DataFrame(eps)
    E.to_csv(f"{OUT}/SPATIAL_EPISODES.csv", index=False)
    cols = ["task", "seed", "config", "arm", "t",
            "exec_dx_m", "exec_dy_m", "exec_dz_m", "exec_rot_x_rad", "exec_rot_y_rad", "exec_rot_z_rad", "exec_gripper",
            "tcp_x_m", "tcp_y_m", "tcp_z_m", "tcp_qw", "tcp_qx", "tcp_qy", "tcp_qz", "gripper_closedness",
            "gen_pred0_dx", "gen_pred0_dy", "gen_pred0_dz", "gen_nat1_token", "ctx_edited_at_step", "n_ensemble_preds",
            "state_hash"]
    T = pd.DataFrame(ts, columns=cols)
    for c in cols:
        if T[c].dtype == np.float64:
            T[c] = T[c].astype(np.float32)
    T.to_parquet(f"{OUT}/SPATIAL_TIME_SERIES.parquet", index=False)
    # ---- paired
    rows = []
    for (task, seed, cfg, arm), a in pos.items():
        for con, (x, y) in (("R-N", ("R", "N")), ("C-F", ("C", "F"))):
            if arm != x:
                continue
            b = pos.get((task, seed, cfg, y))
            if b is None:
                continue
            n = min(len(a["pos"]), len(b["pos"]))
            d = np.linalg.norm(a["pos"][:n] - b["pos"][:n], axis=1)
            rows.append({"task": task, "seed": seed, "config": cfg, "contrast": con, "physical_state_hash": a["state"],
                         "eligible": (a["ok"] is not False) and (b["ok"] is not False),
                         "area_dev_cmstep": 100 * d[8:].sum(), "max_dev_cm": 100 * d.max(), "final_dev_cm": 100 * d[n - 1],
                         "success_treat": a["success"], "success_ref": b["success"], "success_diff": a["success"] - b["success"],
                         **{f"dev_cm_t{t:02d}": 100 * d[t] for t in range(n)}})
    Pd = pd.DataFrame(rows)
    Pd.to_csv(f"{OUT}/tables/V4_paired_units.csv", index=False)
    # ---- success (all eligible episodes)
    el = E[E.eligible & E.arm.isin(["N", "F", "C", "R"])]
    s1 = el.groupby(["config", "arm", "task"]).agg(n=("success", "size"), success_rate=("success", "mean"),
                                                   n_physical_states=("physical_state_id", "nunique")).reset_index()
    s2 = el.groupby(["config", "arm"]).agg(n=("success", "size"), success_rate=("success", "mean"),
                                           n_physical_states=("physical_state_id", "nunique")).reset_index().assign(task="all")
    S = pd.concat([s1, s2], ignore_index=True)
    st = el.groupby(["config", "arm", "task", "physical_state_id"]).success.mean().reset_index()
    s3 = st.groupby(["config", "arm", "task"]).success.mean().rename("success_rate_physical_state_mean").reset_index()
    S = S.merge(s3, on=["config", "arm", "task"], how="left")
    S["note"] = ("move_near: 120 units share 10 physical states; physical-state mean weights each state equally; "
                 "success uses all eligible episodes")
    S.to_csv(f"{OUT}/tables/V4_success_rates.csv", index=False)
    # ---- checks vs stored
    pe = pd.read_csv(f"{V.P1D}/paired_effects.csv")
    cs = pd.read_csv(f"{V.P1D}/condition_summary.csv")
    chk = []
    for cfg in ("r1e1", "r2e1", "r4e1", "r1e4"):
        for con, nm, sg in (("R-N", f"{cfg}:R-N", 1), ("C-F", f"{cfg}:F-C", -1)):
            d = Pd[(Pd.config == cfg) & (Pd.contrast == con) & Pd.eligible]
            for met in ("area_dev_cmstep", "success"):
                r = pe[(pe.name == nm) & (pe.metric == met)]
                mine = d.area_dev_cmstep.mean() if met == "area_dev_cmstep" else d.success_diff.mean()
                stored = float(r.estimate.iloc[0]) * (sg if met == "success" else 1) if len(r) else np.nan
                chk.append({"config": cfg, "contrast": con, "metric": met, "this_package": mine, "stored": stored,
                            "stored_row": nm, "n": len(d), "abs_diff": abs(mine - stored) if len(r) else np.nan,
                            "note": "area is symmetric (|F-C| = |C-F|); success sign flipped for C-F" if con == "C-F" else ""})
    for _, r in cs.iterrows():
        if r.condition in ARMS and ARMS[r.condition] in ("N", "F", "C", "R"):
            m = S[(S.config == r.config) & (S.arm == ARMS[r.condition]) & (S.task == r.task)]
            if len(m):
                chk.append({"config": r.config, "contrast": ARMS[r.condition], "metric": f"success_rate {r.task}",
                            "this_package": float(m.success_rate.iloc[0]), "stored": float(r.success_rate), "n": int(m.n.iloc[0]),
                            "abs_diff": abs(float(m.success_rate.iloc[0]) - float(r.success_rate)), "stored_row": "condition_summary"})
    C = pd.DataFrame(chk)
    C.to_csv(f"{OUT}/tables/V4_checks.csv", index=False)
    print(C.groupby("metric").abs_diff.max())
    sel = select_cases(Pd)
    figures(Pd, S, sel, pos)


def select_cases(Pd):
    cand = Pd[(Pd.config == "r4e1") & Pd.eligible].copy()
    out = []
    for con in ("R-N", "C-F"):
        c = cand[cand.contrast == con]
        for g, gname in ((-1, "worse_success"), (0, "same_success"), (1, "better_success")):
            gd = c[c.success_diff == g].copy()
            if gd.empty:
                out.append({"contrast": con, "group": gname, "n_candidates": 0, "rank": None}); continue
            med = gd.area_dev_cmstep.median()
            gd["_d"] = (gd.area_dev_cmstep - med).abs()
            gd = gd.sort_values(["_d", "task", "seed"])
            first = gd.iloc[0]
            second = gd[(gd.physical_state_hash != first.physical_state_hash) & (gd.task != first.task)]
            if second.empty:
                second = gd[gd.physical_state_hash != first.physical_state_hash]
            picks = [first] + ([second.iloc[0]] if len(second) else [])
            for rk, p in enumerate(picks, 1):
                out.append({"contrast": con, "group": gname, "n_candidates": len(gd), "group_median_area": med, "rank": rk,
                            "task": p.task, "seed": int(p.seed), "physical_state_hash": p.physical_state_hash,
                            "area_dev_cmstep": p.area_dev_cmstep, "success_treat": p.success_treat, "success_ref": p.success_ref,
                            "area_percentile_in_contrast": float((c.area_dev_cmstep <= p.area_dev_cmstep).mean() * 100)})
    S = pd.DataFrame(out)
    S.to_csv(f"{OUT}/tables/V4_selected_cases.csv", index=False)
    cand.groupby(["contrast", "success_diff"]).size().rename("n_units").reset_index().to_csv(f"{OUT}/tables/V4_case_group_counts.csv", index=False)
    print(S.to_string())
    return S


def figures(Pd, S, sel, pos):
    plt = V.mpl_setup()
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.5))
    ax = axs[0]
    tcols = [c for c in Pd.columns if c.startswith("dev_cm_t")]
    for (cfg, con), col, ls in ((("r4e1", "R-N"), V.PAL[0], "-"), (("r1e1", "R-N"), V.PAL[0], (0, (2, 1.5))),
                                (("r4e1", "C-F"), V.PAL[1], "-"), (("r1e1", "C-F"), V.PAL[1], (0, (2, 1.5)))):
        d = Pd[(Pd.config == cfg) & (Pd.contrast == con) & Pd.eligible]
        ax.plot(np.arange(len(tcols)), d[tcols].mean().values, color=col, ls=ls, lw=1.2, label=f"{con} {cfg} (n={len(d)})")
    for t in (8, 12, 16, 20):
        ax.axvline(t, color=V.GRID, lw=0.6)
    ax.set_xlabel("control step t (injections at 8, 12, 16, 20)"); ax.set_ylabel("mean ||TCP_a − TCP_b|| (cm)")
    ax.set_title("Actual TCP translation deviation", loc="left", color=V.INK); ax.legend(frameon=False, fontsize=5.4)
    ax = axs[1]
    d = Pd[(Pd.config == "r4e1") & Pd.eligible]
    for i, (con, col) in enumerate((("R-N", V.PAL[0]), ("C-F", V.PAL[1]))):
        for j, g in enumerate((-1, 0, 1)):
            x = d[(d.contrast == con) & (d.success_diff == g)].area_dev_cmstep
            xpos = j + (i - 0.5) * 0.35
            if len(x):
                ax.boxplot([x], positions=[xpos], widths=0.28, showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor=col, alpha=0.35, edgecolor=col), medianprops=dict(color=V.INK),
                           whiskerprops=dict(color=col), capprops=dict(color=col))
                ax.text(xpos, ax.get_ylim()[1] if False else x.quantile(0.75) * 1.05 + 20, f"n={len(x)}", fontsize=5, ha="center", color=V.INK2)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["success −1", "same", "success +1"], fontsize=6)
    ax.set_ylabel("TCP deviation area (cm·step)"); ax.set_title("r4e1: area by success change", loc="left", color=V.INK)
    from matplotlib.patches import Patch
    ax.legend([Patch(color=V.PAL[0], alpha=0.5), Patch(color=V.PAL[1], alpha=0.5)], ["R−N", "C−F"], frameon=False, fontsize=5.6)
    ax = axs[2]
    arms = ["N", "F", "C", "R"]
    for k, (task, lab) in enumerate((("google_robot_pick_coke_can", "coke"), ("google_robot_move_near", "move_near"))):
        m = S[(S.config == "r4e1") & (S.task == task)].set_index("arm")
        ax.bar(np.arange(4) + (k - 0.5) * 0.38, [100 * m.loc[a, "success_rate"] for a in arms], width=0.36,
               color=V.PAL[2] if k == 0 else V.PAL[6], label=f"{lab} (n={int(m.n.iloc[0])}/arm; "
               f"{int(m.n_physical_states.iloc[0])} physical states)")
    ax.set_xticks(range(4)); ax.set_xticklabels(arms); ax.set_ylabel("success (%)"); ax.set_ylim(0, 100)
    ax.set_title("r4e1 success (all eligible)", loc="left", color=V.INK)
    ax.legend(frameon=False, fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    fig.tight_layout()
    V.save(fig, "V4_spatialvla_tcp_success")
    plt.close(fig)
    # case panels: top-down TCP x/y paths
    sc = sel.dropna(subset=["rank"])
    n = len(sc)
    if n:
        cols = min(n, 4); rws = int(np.ceil(n / cols))
        fig, axs = plt.subplots(rws, cols, figsize=(6.9, 1.9 * rws), squeeze=False)
        for ax, (_, c) in zip(axs.flat, sc.iterrows()):
            x, y = c.contrast.split("-")
            a, b = pos[(c.task, int(c.seed), "r4e1", x)], pos[(c.task, int(c.seed), "r4e1", y)]
            for P, col, lab in ((b["pos"], V.INK2, y), (a["pos"], V.PAL[0] if x == "R" else V.PAL[1], x)):
                ax.plot(P[:, 0], P[:, 1], color=col, lw=1.1, label=f"{lab} (success={int(pos[(c.task, int(c.seed), 'r4e1', lab)]['success'])})")
                ax.plot(P[8, 0], P[8, 1], marker="o", ms=3, color=col)
            ax.set_aspect("equal", adjustable="datalim"); ax.grid(True)
            ax.set_title(f"{c.contrast} · {c.group} · {c.task.replace('google_robot_', '')} s{int(c.seed)}\narea {c.area_dev_cmstep:.0f} cm·step",
                         loc="left", fontsize=5.6, color=V.INK)
            ax.legend(frameon=False, fontsize=5); ax.tick_params(labelsize=5)
        for ax in list(axs.flat)[n:]:
            ax.set_axis_off()
        fig.text(0.01, -0.02, "Actual TCP x/y (m, simulator world frame: sapien tcp link pose) from t = 0 to 79; dot = branch step t = 8. Cases chosen by "
                 "CASE_SELECTION_RULE.md §B; no RGB stored.", fontsize=5.6, color=V.INK2)
        fig.tight_layout()
        V.save(fig, "V4_spatialvla_cases_tcp")
        plt.close(fig)


if __name__ == "__main__":
    main()
