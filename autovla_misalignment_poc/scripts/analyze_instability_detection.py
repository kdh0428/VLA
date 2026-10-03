#!/usr/bin/env python
"""
Instability (amplification) detection baselines, pre- vs post-deviation (P4, outputs/instability_detection_baselines).

Population: candidates that deviate from the GT token at t* < 9 (same as mechanism_selection_link.py), label =
amplification (A- and FDE5 > 3 m). Every detector has a pre-deviation version (information up to and including the
step that emits t*) and a post-deviation version (steps t*+1 .. 9), with a sign fixed in advance (no tuning):
  entropy          mean decoding entropy                         higher = unstable
  loglik           summed log-prob of the emitted tokens          lower  = unstable
  margin           mean top-1 minus top-2 probability             lower  = unstable    (teacher-forced pass)
  cand_variance    spread of the 17 candidates' poses (scene-level)    higher = unstable
  disagreement     mean distance of this candidate to the other 16      higher = unstable
  medoid_dist      distance of this candidate to the scene medoid       higher = unstable
  (pre: poses 0 .. t*, post: poses t*+1 .. 9)
  probe_logreg / probe_ridge / probe_mlp   hidden-state detectors (layer 18 or 36, pre = state emitting t*,
                   post = mean state over t*+1 .. 9), trained on dev only (hyper-parameters and layer chosen by
                   log-grouped 3-fold CV on dev), evaluated once on held-out.
Metrics (held-out; dev for reference): pooled AUROC and AUPRC with log-cluster bootstrap 95% CI (1,000), and the
within-scene AUROC (mean over scenes containing both classes), as in mechanism_selection_link.py.

  python scripts/analyze_instability_detection.py <out_dir> --dev RUN:TF [RUN:TF ...] --heldout RUN:TF [...]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_action_history import AMP_FDE, a_eval   # noqa: E402

N_ACT, HID = 10, 2048
SIGNS = {"entropy": 1, "loglik": -1, "margin": -1, "cand_variance": 1, "disagreement": 1, "medoid_dist": 1}


def load_set(pairs, layers):
    rows, H = [], {(w, l): [] for w in ("pre", "post") for l in layers}
    for pair in pairs:
        run, tf = pair.split(":")
        steps = {r["token"]: r for r in map(json.loads, open(os.path.join(tf, "steps.jsonl")))}
        idx = [json.loads(l) for l in open(os.path.join(tf, "hidden_index.jsonl"))]
        mm = {k: np.memmap(os.path.join(tf, f"hidden_{k[0]}_L{k[1]}.f16"), dtype=np.float16, mode="r").reshape(-1, HID)
              for k in H}
        pos = {(x["token"], x["cand"]): i for i, x in enumerate(idx)}
        for r in map(json.loads, open(os.path.join(run, "records.jsonl"))):
            if r.get("stub_cot") or r["token"] not in steps:
                continue
            st = steps[r["token"]]; C = r["candidates"]; gt = r["gt"]
            gxy = np.asarray(r["trajectory_gt"], float)[:, :2]
            T = np.array([np.asarray(c["trajectory_pred"], float) for c in C])            # (17, 10, 2)
            D = np.linalg.norm(T[:, None] - T[None], axis=-1)                               # (17, 17, 10)
            med = int(np.argmin(D.mean(-1).sum(1)))
            for i, c in enumerate(C):
                ts = next((k for k in range(N_ACT) if c["action_idx"][k] != gt[k]), None)
                if ts is None or ts >= N_ACT - 1 or (r["token"], i) not in pos:
                    continue
                err = np.linalg.norm(T[i] - gxy, axis=1)
                amp = float((not a_eval(T[i].tolist(), gxy.tolist())) and err[-1] > AMP_FDE)
                pre, post = slice(0, ts + 1), slice(ts + 1, N_ACT)
                f = {}
                for nm, sl in (("pre", pre), ("post", post)):
                    f[f"entropy_{nm}"] = float(np.mean(c["entropy_steps"][sl]))
                    f[f"loglik_{nm}"] = float(np.sum(c["logprob_steps"][sl]))
                    f[f"margin_{nm}"] = float(np.mean(st["margin"][i][sl]))
                    f[f"cand_variance_{nm}"] = float(T[:, sl].var(0).sum(-1).mean())
                    f[f"disagreement_{nm}"] = float(np.delete(D[i, :, sl], i, 0).mean())
                    f[f"medoid_dist_{nm}"] = float(D[i, med, sl].mean())
                rows.append({"token": r["token"], "log": r["log"], "cand": i, "t_star": ts, "amp": amp, **f})
                j = pos[(r["token"], i)]
                for k in H:
                    H[k].append(np.asarray(mm[k][j], np.float32))
    return rows, {k: np.stack(v) for k, v in H.items()}


def metrics(rows, score, reps=1000):
    y = np.array([r["amp"] for r in rows]); s = np.asarray(score, float)
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["log"]].append(i)
    logs = sorted(by); rng = random.Random(0)
    au, ap = [], []
    for _ in range(reps):
        ii = np.concatenate([by[rng.choice(logs)] for _ in logs])
        if len(set(y[ii])) == 2:
            au.append(roc_auc_score(y[ii], s[ii])); ap.append(average_precision_score(y[ii], s[ii]))
    sc = defaultdict(list)
    for i, r in enumerate(rows):
        sc[r["token"]].append(i)
    within = [roc_auc_score(y[v], s[v]) for v in sc.values() if len(set(y[v])) == 2]
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]   # noqa: E731
    return {"auroc": float(roc_auc_score(y, s)), "auroc_ci95": q(au), "auprc": float(average_precision_score(y, s)),
            "auprc_ci95": q(ap), "prevalence": float(y.mean()), "within_scene_auroc": float(np.mean(within)),
            "n_mixed_scenes": len(within), "n": len(rows)}


def probes(rows_d, H_d, rows_h, H_h, layers):
    y_d = np.array([r["amp"] for r in rows_d]); g_d = np.array([r["log"] for r in rows_d])
    makers = {"probe_logreg": [(f"C={C}", lambda C=C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)))
                               for C in (1e-3, 1e-2)],
              "probe_ridge": [(f"alpha={al}", lambda al=al: make_pipeline(StandardScaler(), RidgeClassifier(alpha=al)))
                              for al in (1e3, 1e4)],
              "probe_mlp": [(f"alpha={al}", lambda al=al: make_pipeline(StandardScaler(), MLPClassifier(
                  hidden_layer_sizes=(256,), alpha=al, early_stopping=True, max_iter=200, random_state=0))) for al in (1e-2,)]}
    out = {}
    for name, grid in makers.items():
        for w in ("pre", "post"):
            best = None
            for l in layers:
                for hp, mk in grid:
                    s = np.zeros(len(y_d))
                    for tr, te in GroupKFold(3).split(H_d[(w, l)], y_d, g_d):
                        m = mk().fit(H_d[(w, l)][tr], y_d[tr])
                        s[te] = m.decision_function(H_d[(w, l)][te]) if hasattr(m, "decision_function") else m.predict_proba(H_d[(w, l)][te])[:, 1]
                    a = roc_auc_score(y_d, s)
                    if best is None or a > best[0]:
                        best = (a, l, hp, mk)
            a, l, hp, mk = best
            m = mk().fit(H_d[(w, l)], y_d)
            sh = m.decision_function(H_h[(w, l)]) if hasattr(m, "decision_function") else m.predict_proba(H_h[(w, l)])[:, 1]
            out[f"{name}_{w}"] = {"dev_cv_auroc": float(a), "layer": l, "hparam": hp, "heldout_scores": sh}
            print(f"  {name}_{w}: layer {l}, {hp}, dev CV AUROC {a:.3f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--heldout", nargs="+", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[18, 36])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows_d, H_d = load_set(a.dev, a.layers); rows_h, H_h = load_set(a.heldout, a.layers)
    print(f"dev {len(rows_d)} deviating candidates ({np.mean([r['amp'] for r in rows_d]):.3f} amplifying), "
          f"held-out {len(rows_h)} ({np.mean([r['amp'] for r in rows_h]):.3f})", flush=True)
    res = {"dev": {}, "heldout": {}}
    for det, sg in SIGNS.items():
        for w in ("pre", "post"):
            k = f"{det}_{w}"
            res["dev"][k] = metrics(rows_d, [sg * r[k] for r in rows_d])
            res["heldout"][k] = metrics(rows_h, [sg * r[k] for r in rows_h])
    pr = probes(rows_d, H_d, rows_h, H_h, a.layers)
    for k, v in pr.items():
        res["heldout"][k] = {**metrics(rows_h, v["heldout_scores"]), "layer": v["layer"], "hparam": v["hparam"],
                             "dev_cv_auroc": v["dev_cv_auroc"]}
    json.dump(res, open(os.path.join(a.out, "detection.json"), "w"), indent=1)
    L = ["| detector | pre AUROC [95% CI] | pre AUPRC | pre within-scene | post AUROC [95% CI] | post AUPRC | post within-scene |",
         "|---|---|---:|---:|---|---:|---:|"]
    for det in list(SIGNS) + ["probe_logreg", "probe_ridge", "probe_mlp"]:
        c = []
        for w in ("pre", "post"):
            m = res["heldout"][f"{det}_{w}"]
            c.append(f"{m['auroc']:.3f} [{m['auroc_ci95'][0]:.3f}, {m['auroc_ci95'][1]:.3f}] | {m['auprc']:.3f} | {m['within_scene_auroc']:.3f}")
        L.append(f"| {det} | " + " | ".join(c) + " |")
    p = res["heldout"]["entropy_pre"]["prevalence"]
    txt = f"held-out: {len(rows_h)} deviating candidates, amplification prevalence {p:.3f} (AUPRC of a random detector)\n\n" + "\n".join(L) + "\n"
    open(os.path.join(a.out, "detection.md"), "w").write(txt); print(txt)


if __name__ == "__main__":
    main()
