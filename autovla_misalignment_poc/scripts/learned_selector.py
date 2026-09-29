#!/usr/bin/env python
"""
Learned best-of-N selectors (CPU only).

  dev mode   : python scripts/learned_selector.py cv <features.npz> <out_dir>
               log-grouped 5-fold cross-validation of every selector on the dev set; saves the
               comparison and fits the final models on the whole dev set (<out_dir>/models.json)
  test mode  : python scripts/learned_selector.py test <features.npz> <out_dir>
               applies the frozen models from <out_dir>/models.json to a held-out feature set

Selectors (features z-scored across the candidates of each scene; is_natural kept 0/1):
  normal, max_loglik, min_entropy, ranksum          fixed rules (experiments 18-20)
  ridge_fde        Ridge regression of within-scene centred log(FDE5 + 0.1); pick the lowest prediction
  logit_aplus      logistic regression of A+ (candidate level); pick the highest probability
  ridge_fde_gated  ridge_fde, but keep the natural plan unless its predicted log-FDE exceeds the chosen
                   candidate's by more than delta (delta tuned inside the training folds, cost = A- + 0.1 FDE)
Metrics: population A- rate and FDE5, paired vs normal, log-cluster bootstrap CI, McNemar.
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold

DELTAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def zscene(X):
    """z-score each feature across candidates within a scene; keep is_natural (index 8) raw."""
    mu = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True) + 1e-6
    Z = (X - mu) / sd
    Z[:, :, 8] = X[:, :, 8]
    return Z


def fixed_pick(X, rule):
    if rule == "normal":
        return np.zeros(len(X), int)
    if rule == "max_loglik":
        return X[:, :, 3].argmax(1)
    if rule == "min_entropy":
        return X[:, :, 0].argmin(1)
    if rule == "ranksum":
        out = []
        for s in X:
            re_ = rankdata(s[:, 0]); rl_ = rankdata(-s[:, 3])
            out.append(int(np.argmin(re_ + rl_ + 1e-6 * re_)))
        return np.asarray(out)
    raise ValueError(rule)


def fit(Z, A, F):
    n, c, f = Z.shape
    y = np.log(F + 0.1); y = y - y.mean(1, keepdims=True)
    ridge = Ridge(alpha=1.0).fit(Z.reshape(-1, f), y.reshape(-1))
    logit = LogisticRegression(max_iter=2000, C=1.0).fit(Z.reshape(-1, f), (1 - A).reshape(-1))
    return ridge, logit


def pick_models(Z, ridge, logit, delta):
    n, c, f = Z.shape
    pf = ridge.predict(Z.reshape(-1, f)).reshape(n, c)
    pa = logit.predict_proba(Z.reshape(-1, f))[:, 1].reshape(n, c)
    best = pf.argmin(1)
    gated = np.where(pf[np.arange(n), 0] - pf[np.arange(n), best] > delta, best, 0)
    return {"ridge_fde": best, "logit_aplus": pa.argmax(1), "ridge_fde_gated": gated}


def evaluate(picks, A, F, logs, reps=2000):
    idx = np.arange(len(A))
    a0, f0 = A[idx, 0], F[idx, 0]
    by = defaultdict(list)
    for i, lg in enumerate(logs):
        by[lg].append(i)
    L = sorted(by)
    rng = random.Random(0)
    boots = [np.concatenate([by[rng.choice(L)] for _ in L]) for _ in range(reps)]
    out = {}
    for name, p in picks.items():
        a, f = A[idx, p], F[idx, p]
        da, df = a - a0, f - f0
        gain, loss = int(((a == 1) & (a0 == 0)).sum()), int(((a == 0) & (a0 == 1)).sum())
        out[name] = {"a_minus": float(a.mean()), "fde5": float(f.mean()),
                     "d_a_minus": float(da.mean()), "d_a_minus_ci95": [float(np.percentile([da[b].mean() for b in boots], q)) for q in (2.5, 97.5)],
                     "d_fde5": float(df.mean()), "d_fde5_ci95": [float(np.percentile([df[b].mean() for b in boots], q)) for q in (2.5, 97.5)],
                     "mcnemar": {"new_failures": gain, "rescued": loss, "p": 1.0 if gain + loss == 0 else float(binomtest(gain, gain + loss, 0.5).pvalue)},
                     "switched_from_natural": float((p != 0).mean())}
    return out


def table(res, title):
    print(f"\n{title}")
    print(f"{'selector':16s} {'A- %':>6s} {'dA- %p [CI]':>24s} {'new/resc':>9s} {'McN p':>8s} {'FDE5':>6s} {'dFDE [CI]':>24s} {'switch':>7s}")
    for k, v in res.items():
        print(f"{k:16s} {100*v['a_minus']:6.2f} {100*v['d_a_minus']:+6.2f} [{100*v['d_a_minus_ci95'][0]:+.2f},{100*v['d_a_minus_ci95'][1]:+.2f}] "
              f"{v['mcnemar']['new_failures']:4d}/{v['mcnemar']['rescued']:<4d} {v['mcnemar']['p']:8.2g} {v['fde5']:6.3f} "
              f"{v['d_fde5']:+.3f} [{v['d_fde5_ci95'][0]:+.3f},{v['d_fde5_ci95'][1]:+.3f}] {100*v['switched_from_natural']:6.1f}%")


def main() -> None:
    mode, path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    d = np.load(path)
    X, A, F, logs = d["X"].astype(np.float64), d["A"], d["FDE"], d["logs"]
    Z = zscene(X)
    fixed = {r: fixed_pick(X, r) for r in ("normal", "max_loglik", "min_entropy", "ranksum")}
    if mode == "cv":
        oof = {k: np.zeros(len(X), int) for k in ("ridge_fde", "logit_aplus", "ridge_fde_gated")}
        chosen = []
        for tr, te in GroupKFold(5).split(Z, groups=logs):
            ridge, logit = fit(Z[tr], A[tr], F[tr])
            # tune the gate on the training folds only
            cost = {dl: (A[tr, pick_models(Z[tr], ridge, logit, dl)["ridge_fde_gated"]].mean()
                         + 0.1 * F[tr, pick_models(Z[tr], ridge, logit, dl)["ridge_fde_gated"]].mean()) for dl in DELTAS}
            dl = min(cost, key=cost.get); chosen.append(dl)
            p = pick_models(Z[te], ridge, logit, dl)
            for k in oof:
                oof[k][te] = p[k][np.arange(len(te))]
        res = evaluate({**fixed, **oof}, A, F, logs)
        table(res, f"DEV (log-grouped 5-fold CV, {len(X)} scenes, {len(set(logs))} logs); gate delta per fold {chosen}")
        ridge, logit = fit(Z, A, F)
        cost = {dl: (A[np.arange(len(A)), pick_models(Z, ridge, logit, dl)["ridge_fde_gated"]].mean()
                     + 0.1 * F[np.arange(len(A)), pick_models(Z, ridge, logit, dl)["ridge_fde_gated"]].mean()) for dl in DELTAS}
        dl = min(cost, key=cost.get)
        feats = [str(x) for x in d["feats"]]
        models = {"feats": feats, "delta": dl,
                  "ridge": {"coef": ridge.coef_.tolist(), "intercept": float(ridge.intercept_)},
                  "logit": {"coef": logit.coef_[0].tolist(), "intercept": float(logit.intercept_[0])}}
        json.dump(models, open(os.path.join(out_dir, "models.json"), "w"), indent=1)
        json.dump(res, open(os.path.join(out_dir, "dev_cv.json"), "w"), indent=1)
        print("\nfinal ridge coefficients:", {f: round(c, 3) for f, c in zip(feats, ridge.coef_)}, " delta", dl)
    else:
        m = json.load(open(os.path.join(out_dir, "models.json")))
        ridge = Ridge(); ridge.coef_ = np.asarray(m["ridge"]["coef"]); ridge.intercept_ = m["ridge"]["intercept"]
        logit = LogisticRegression(); logit.coef_ = np.asarray([m["logit"]["coef"]]); logit.intercept_ = np.asarray([m["logit"]["intercept"]])
        logit.classes_ = np.asarray([0.0, 1.0])
        res = evaluate({**fixed, **pick_models(Z, ridge, logit, m["delta"])}, A, F, logs)
        table(res, f"HELD-OUT ({len(X)} scenes, {len(set(logs))} logs), frozen models, delta {m['delta']}")
        json.dump(res, open(os.path.join(out_dir, "heldout.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
