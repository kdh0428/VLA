#!/usr/bin/env python
"""G (post_hoc/exploratory CPU pilot): can the gain of replacing the natural plan (keep) by a runtime GT-free candidate
(rank-sum pick among 16 T=1.0 samples) be predicted before execution?  Gates differ only in the trigger.
Development = expanded_best_of_n (navtest shards 6-17; log-grouped 5-fold CV for fitting/thresholds);
heldout = heldout_best_of_n (shards 18-31; evaluated once with frozen models). Open-loop FDE5 only.
enable_new_inference = False: this script never runs a model.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ra_common as R  # noqa: E402
from analyze_action_history import a_eval  # noqa: E402

ENABLE_NEW_INFERENCE = False
G = f"{R.OUT}/selective_intervention"
DEV = [f"{R.POC}/outputs/expanded_best_of_n/gpu0/records.jsonl", f"{R.POC}/outputs/expanded_best_of_n/gpu1/records.jsonl"]
HO = [f"{R.POC}/outputs/heldout_best_of_n/gpu0/records.jsonl", f"{R.POC}/outputs/heldout_best_of_n/gpu1/records.jsonl"]
RATES = [0.05, 0.10, 0.20, 0.30, 0.50]


def cand_feats(c, trajs, i):
    e = np.asarray(c["entropy_steps"], float); lp = np.asarray(c["logprob_steps"], float)
    T = trajs[i]
    others = np.delete(np.arange(len(trajs)), i)
    cons = np.linalg.norm(trajs[others] - T[None], axis=2).mean()
    d = np.diff(np.vstack([[0, 0], T]), axis=0)
    return {"ent_mean": e.mean(), "ent_max": e.max(), "ent_late": e[5:].mean(), "lp_sum": lp.sum(), "lp_min": lp.min(),
            "cons_ade": cons, "path_len": np.linalg.norm(d, axis=1).sum(), "final_lat": T[-1, 1]}


def load(paths, split):
    rows, cands = [], []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            c = r["candidates"]
            if len(c) < 2:
                continue
            trajs = np.array([np.asarray(x["trajectory_pred"], float)[:, :2] for x in c])
            gt = np.asarray(r["trajectory_gt"], float)[:, :2]
            fde = np.linalg.norm(trajs[:, -1] - gt[-1], axis=1)
            re_ = rankdata([np.mean(x["entropy_steps"]) for x in c]); rl_ = rankdata([-sum(x["logprob_steps"]) for x in c])
            rs = int(np.argmin(re_ + rl_ + 1e-6 * re_))
            fk = cand_feats(c[0], trajs, 0); fr = cand_feats(c[rs], trajs, rs)
            keep_ok = a_eval(trajs[0].tolist(), gt.tolist())
            rs_ok = keep_ok if rs == 0 else a_eval(trajs[rs].tolist(), gt.tolist())
            spread = float(np.mean([np.linalg.norm(trajs[1:] - trajs[j][None], axis=2).mean() for j in range(1, len(trajs))]))
            row = {"split": split, "group_id": r["log"], "scene_id": r["token"], "drive_id": R.drive_of(r["log"]),
                   "rs_index": rs, "rs_is_keep": rs == 0, "fde_keep": fde[0], "fde_rs": fde[rs],
                   "severe_keep": int((not keep_ok) and fde[0] > 3), "severe_rs": int((not rs_ok) and fde[rs] > 3),
                   "gain_rs": fde[0] - fde[rs], "fde_oracle": fde.min(), "spread": spread,
                   "ade_keep_rs": float(np.linalg.norm(trajs[0] - trajs[rs], axis=1).mean())}
            row.update({f"keep_{k}": v for k, v in fk.items()}); row.update({f"rs_{k}": v for k, v in fr.items()})
            row.update({f"diff_{k}": fk[k] - fr[k] for k in fk})
            rows.append(row)
            for j in range(len(c)):
                f = cand_feats(c[j], trajs, j)
                cands.append({"split": split, "scene_id": r["token"], "group_id": r["log"], "candidate_id": j,
                              "candidate_source": "natural T=0.01 (keep)" if j == 0 else "sample T=1.0", "is_keep": int(j == 0),
                              "fde": fde[j], **f, "ade_to_keep": float(np.linalg.norm(trajs[j] - trajs[0], axis=1).mean())})
    return pd.DataFrame(rows), pd.DataFrame(cands)


FEAT = None


def main():
    global FEAT
    assert not ENABLE_NEW_INFERENCE
    dev, devc = load(DEV, "development")
    ho, hoc = load(HO, "heldout")
    FEAT = [c for c in dev.columns if c.startswith(("keep_", "rs_", "diff_")) and c != "rs_index" and c != "rs_is_keep"] + ["spread", "ade_keep_rs"]
    runtime = pd.concat([dev, ho])[["split", "group_id", "scene_id"] + FEAT]
    runtime.to_parquet(f"{G}/G_runtime_features.parquet", index=False)        # no outcome columns
    outcome = pd.concat([dev, ho])[["split", "group_id", "scene_id", "rs_index", "rs_is_keep", "fde_keep", "fde_rs", "gain_rs",
                                    "severe_keep", "severe_rs", "fde_oracle"]]
    outcome.to_parquet(f"{G}/G_evaluation_outcomes.parquet", index=False)      # evaluation only
    leak = [c for c in FEAT if any(k in c for k in ("fde", "severe", "gain", "gt", "oracle"))]
    assert not leak, leak
    Xd, Xh = dev[FEAT].values, ho[FEAT].values
    gk = GroupKFold(5)
    oof = {"gain": np.zeros(len(dev)), "risk": np.zeros(len(dev))}
    for tr, te in gk.split(Xd, groups=dev.group_id):
        m1 = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xd[tr], dev.gain_rs.values[tr])
        oof["gain"][te] = m1.predict(Xd[te])
        m2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xd[tr], dev.severe_keep.values[tr])
        oof["risk"][te] = m2.predict_proba(Xd[te])[:, 1]
    mg = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xd, dev.gain_rs.values)
    mr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xd, dev.severe_keep.values)
    score_dev = {"gain": oof["gain"], "risk": oof["risk"], "entropy": dev.keep_ent_mean.values}
    score_ho = {"gain": mg.predict(Xh), "risk": mr.predict_proba(Xh)[:, 1], "entropy": ho.keep_ent_mean.values}
    # candidate critic (best-of-k, no gate): ridge on candidate runtime features -> log(FDE+0.1), within-scene argmin
    cf = ["is_keep", "ent_mean", "ent_max", "ent_late", "lp_sum", "lp_min", "cons_ade", "path_len", "final_lat", "ade_to_keep"]
    crit = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(devc[cf].values, np.log(devc.fde.values + 0.1))
    hoc["pred"] = crit.predict(hoc[cf].values)
    pick = hoc.loc[hoc.groupby("scene_id").pred.idxmin()].set_index("scene_id")
    res = []

    def evaluate(df, trig, name, rate_target, thr):
        fde_new = np.where(trig, df.fde_rs, df.fde_keep)
        sev_new = np.where(trig, df.severe_rs, df.severe_keep)
        fail = df.severe_keep.values == 1
        return {"gate": name, "target_rate": rate_target, "threshold_from_dev": thr, "realised_rate": float(trig.mean()),
                "effective_edit_rate": float((trig & ~df.rs_is_keep.values).mean()),
                "n_scenes": len(df), "n_groups": df.group_id.nunique(),
                "delta_mean_fde_m": float((fde_new - df.fde_keep).mean()),
                "delta_severe_pp": float(100 * (sev_new - df.severe_keep).mean()),
                "harm_keep_ok_to_severe": int(((df.severe_keep == 0) & (sev_new == 1)).sum()),
                "benefit_keep_severe_to_ok": int(((df.severe_keep == 1) & (sev_new == 0)).sum()),
                "fde_worse_by_1m": int(((fde_new - df.fde_keep) > 1).sum()), "fde_better_by_1m": int(((fde_new - df.fde_keep) < -1).sum()),
                "failure_subset_delta_fde_m": float((fde_new - df.fde_keep)[fail].mean()) if fail.any() else np.nan,
                "failure_subset_n": int(fail.sum())}
    for split, df, S, Sd in (("development (out-of-fold)", dev, score_dev, score_dev), ("heldout (frozen)", ho, score_ho, score_dev)):
        res.append({"split": split, **evaluate(df, np.zeros(len(df), bool), "keep (no edit)", 0.0, np.nan)})
        res.append({"split": split, **evaluate(df, np.ones(len(df), bool), "always rank-sum", 1.0, np.nan)})
        for rt in RATES:
            for gname in ("gain", "risk", "entropy"):
                thr = float(np.quantile(Sd[gname], 1 - rt))
                res.append({"split": split, **evaluate(df, S[gname] > thr, f"{gname} gate", rt, thr)})
            # random gate: exact expectation over Bernoulli(rt) triggers
            ex = evaluate(df, np.ones(len(df), bool), "random gate (expectation)", rt, np.nan)
            for k in ("delta_mean_fde_m", "delta_severe_pp", "failure_subset_delta_fde_m"):
                ex[k] *= rt
            for k in ("harm_keep_ok_to_severe", "benefit_keep_severe_to_ok", "fde_worse_by_1m", "fde_better_by_1m"):
                ex[k] = ex[k] * rt
            ex["realised_rate"] = rt
            ex["effective_edit_rate"] = rt * float((~df.rs_is_keep.values).mean())
            res.append({"split": split, **ex})
    hk = ho.set_index("scene_id")
    fde_c = pick.fde.reindex(hk.index).values
    res.append({"split": "heldout (frozen)", "gate": "candidate critic (best-of-17, no gate)", "target_rate": np.nan,
                "realised_rate": float((pick.candidate_id.reindex(hk.index) != 0).mean()), "n_scenes": len(hk),
                "delta_mean_fde_m": float((fde_c - hk.fde_keep.values).mean())})
    res.append({"split": "heldout (frozen)", "gate": "oracle min-FDE (not deployable)", "target_rate": 1.0,
                "delta_mean_fde_m": float((ho.fde_oracle - ho.fde_keep).mean()), "n_scenes": len(ho)})
    Rz = pd.DataFrame(res)
    Rz["status"] = "post_hoc/exploratory; open-loop FDE5; GT used only for training labels/evaluation"
    Rz.to_csv(f"{G}/G_gate_results.csv", index=False)
    # calibration of predicted gain (heldout)
    cal = pd.DataFrame({"pred": score_ho["gain"], "obs": ho.gain_rs.values})
    cal["decile"] = pd.qcut(cal.pred, 10, labels=False, duplicates="drop")
    cal.groupby("decile").agg(pred_mean=("pred", "mean"), obs_mean=("obs", "mean"), n=("obs", "size")).to_csv(f"{G}/G_gain_calibration_heldout.csv")
    # row-level table in requested format
    rows = []
    for split, df, S in (("development", dev, score_dev), ("heldout", ho, score_ho)):
        rows.append(pd.DataFrame({"group_id": df.group_id, "split": split, "scene_id": df.scene_id, "candidate_id": df.rs_index,
                                  "candidate_source": "rank-sum pick among natural + 16 T=1.0 samples",
                                  "runtime_feature_availability": "all runtime (entropy/log-prob/consensus/geometry)",
                                  "loss_keep": df.fde_keep, "loss_edit": df.fde_rs, "observed_gain": df.gain_rs,
                                  "predicted_gain": S["gain"], "inference_cost_status": "estimated: 17 decodes per scene (1 + 16), "
                                  "no wall-clock measured here"}))
    pd.concat(rows).to_csv(f"{G}/G_rowlevel.csv.gz", index=False, compression="gzip")
    inv = {"development_scenes": len(dev), "heldout_scenes": len(ho), "dev_logs": int(dev.group_id.nunique()),
           "heldout_logs": int(ho.group_id.nunique()), "rs_is_keep_share_dev": float(dev.rs_is_keep.mean()),
           "rs_is_keep_share_heldout": float(ho.rs_is_keep.mean()),
           "dev_heldout_shared_days_vehicles": sorted(set(d.split("_")[0][:10] + "_" + d.split("_")[1] for d in dev.group_id) &
                                                      set(d.split("_")[0][:10] + "_" + d.split("_")[1] for d in ho.group_id))[:50]}
    json.dump(inv, open(f"{G}/G_inventory_counts.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(json.dumps({k: v for k, v in inv.items() if k != "dev_heldout_shared_days_vehicles"}), len(inv["dev_heldout_shared_days_vehicles"]))
    print(Rz[["split", "gate", "target_rate", "realised_rate", "delta_mean_fde_m", "delta_severe_pp", "harm_keep_ok_to_severe",
              "benefit_keep_severe_to_ok"]].round(4).to_string())
    figure(Rz)


def figure(Rz):
    plt = R.mpl()
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.4))
    h = Rz[Rz.split == "heldout (frozen)"]
    for gname, col, mk in (("gain gate", R.PAL[0], "o"), ("risk gate", R.PAL[1], "s"), ("entropy gate", R.PAL[2], "^"),
                           ("random gate (expectation)", R.INK2, "x")):
        d = h[h.gate == gname].sort_values("target_rate")
        axs[0].plot(100 * d.realised_rate, d.delta_mean_fde_m, marker=mk, color=col, ms=3.5, label=gname)
        axs[1].plot(100 * d.realised_rate, d.delta_severe_pp, marker=mk, color=col, ms=3.5)
    a = h[h.gate == "always rank-sum"].iloc[0]
    axs[0].plot([100], [a.delta_mean_fde_m], "D", color=R.PAL[6], ms=4, label="always rank-sum")
    axs[1].plot([100], [a.delta_severe_pp], "D", color=R.PAL[6], ms=4)
    for ax in axs:
        ax.axhline(0, color=R.INK2, lw=0.6); ax.grid(True); ax.set_xlabel("intervention rate on heldout (%)")
    axs[0].set_ylabel("Δ mean FDE5 vs keep (m)"); axs[1].set_ylabel("Δ severe events vs keep (pp)")
    axs[0].legend(frameon=False, fontsize=5.5); axs[0].set_title("Heldout, frozen thresholds (open-loop)", loc="left")
    fig.text(0.01, -0.05, "All gates apply the same rank-sum candidate; only the trigger differs. Development/heldout share days and vehicles "
             "(not an independent transfer test). Post-hoc exploratory pilot.", fontsize=5.6, color=R.INK2)
    fig.tight_layout()
    R.save(fig, "G_rate_utility", sub="selective_intervention")


if __name__ == "__main__":
    main()
