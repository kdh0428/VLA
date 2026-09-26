"""
Failure-conditioned representation analysis.

The comparison that matters is NOT success-vs-failure on average (that is already known to
work and is explicitly excluded as a contribution). It is:

    control        P+R+A+   (perception right, reasoning right, action right)
    main failure   P+R+A-   (perception right, reasoning right, action WRONG)
    contrast       P+R-A-   (reasoning already wrong)

Both groups have correct observable perception, so any layer-wise divergence between them
cannot be explained by the scene being misperceived.

CONFOUND CONTROL (spec section 10)
A geometric score can be a re-description of the model's own output error. Two guards:
  * `matched_groups` pairs each failure with a control frame from a DIFFERENT clip that has
    the same scenario and the same GT action semantics, so group differences are not just
    "these are different driving situations";
  * `divergence_vs_error_correlation` reports how strongly the per-layer geometric score
    tracks ADE. A score that correlates ~1.0 with ADE at every layer is measuring the
    error, not a representational property, and is reported as such.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def _unit(a: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, eps)


def cosine_to_centroid(X: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    return _unit(X) @ _unit(centroid.reshape(1, -1)).ravel()


def group_stats(X_ctrl: np.ndarray, X_fail: np.ndarray) -> dict:
    """
    Per-layer comparison of a control and a failure group.

    `centroid_cosine` is the control centroid direction vs each group's mean; the reported
    `separation` is the AUROC of a single-feature score (cosine to the control centroid),
    which is a scale-free measure of how distinguishable the two groups are at this layer.
    """
    out: dict[str, float] = {}
    if len(X_ctrl) == 0 or len(X_fail) == 0:
        return {"n_ctrl": len(X_ctrl), "n_fail": len(X_fail)}

    c_ctrl = X_ctrl.mean(axis=0)
    c_fail = X_fail.mean(axis=0)

    out["n_ctrl"] = int(len(X_ctrl))
    out["n_fail"] = int(len(X_fail))
    out["centroid_cosine"] = float(_unit(c_ctrl.reshape(1, -1)) @ _unit(c_fail.reshape(1, -1)).T)
    out["centroid_l2"] = float(np.linalg.norm(c_ctrl - c_fail))
    out["norm_ctrl"] = float(np.linalg.norm(X_ctrl, axis=1).mean())
    out["norm_fail"] = float(np.linalg.norm(X_fail, axis=1).mean())

    # Within-group spread, to tell a real shift from two clouds that simply overlap.
    sd = np.concatenate([X_ctrl - c_ctrl, X_fail - c_fail]).std()
    out["pooled_std"] = float(sd)
    out["centroid_l2_normalised"] = float(out["centroid_l2"] / max(sd, 1e-8))

    s_ctrl = cosine_to_centroid(X_ctrl, c_ctrl)
    s_fail = cosine_to_centroid(X_fail, c_ctrl)
    out["cos_to_ctrl_centroid_mean_ctrl"] = float(s_ctrl.mean())
    out["cos_to_ctrl_centroid_mean_fail"] = float(s_fail.mean())

    # AUROC of that one score, computed without sklearn (rank statistic).
    scores = np.concatenate([s_ctrl, s_fail])
    labels = np.concatenate([np.ones(len(s_ctrl)), np.zeros(len(s_fail))])
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    n1 = labels.sum()
    n0 = len(labels) - n1
    if n1 > 0 and n0 > 0:
        out["separation_auroc"] = float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return out


def matched_groups(rows_ctrl: list[dict], rows_fail: list[dict],
                   keys: tuple[str, ...] = ("scenario", "high_level_action_gt"),
                   seed: int = 0) -> tuple[list[dict], list[dict]]:
    """
    Match control frames to failure frames on `keys`, forbidding same-clip pairs.

    Returns (matched_ctrl, matched_fail) of equal length. Matching removes the trivial
    explanation that the two groups simply contain different driving situations.
    """
    rng = np.random.default_rng(seed)
    pool: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows_ctrl:
        pool[tuple(r.get(k) for k in keys)].append(r)
    for v in pool.values():
        rng.shuffle(v)

    mc, mf = [], []
    used: set[str] = set()
    for f in rows_fail:
        k = tuple(f.get(key) for key in keys)
        for cand in pool.get(k, []):
            if cand["sample_id"] in used or cand["clip_id"] == f["clip_id"]:
                continue
            used.add(cand["sample_id"])
            mc.append(cand)
            mf.append(f)
            break
    return mc, mf


def divergence_vs_error_correlation(scores: np.ndarray, ade: np.ndarray) -> dict:
    """
    Is the per-layer geometric score just a restatement of trajectory error?

    Reports Pearson and Spearman correlation between the score and ADE. |rho| close to 1
    means the score carries no information beyond the error it is supposed to explain.
    """
    m = np.isfinite(scores) & np.isfinite(ade)
    if m.sum() < 8:
        return {"n": int(m.sum())}
    s, a = scores[m], ade[m]
    out = {"n": int(m.sum())}
    if s.std() > 0 and a.std() > 0:
        out["pearson"] = float(np.corrcoef(s, a)[0, 1])
        rs = np.argsort(np.argsort(s)).astype(float)
        ra = np.argsort(np.argsort(a)).astype(float)
        out["spearman"] = float(np.corrcoef(rs, ra)[0, 1])
    return out


def pca_project(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Plain PCA (visualisation aid only, never the main evidence)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comp = Vt[:n_components]
    var = (S ** 2) / max(len(X) - 1, 1)
    ratio = var[:n_components] / var.sum() if var.sum() > 0 else np.zeros(n_components)
    return Xc @ comp.T, ratio


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two representation sets over the same samples."""
    if len(X) != len(Y) or len(X) < 3:
        return float("nan")
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(Xc.T @ Yc, "fro") ** 2
    den = np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro")
    return float(num / den) if den > 0 else float("nan")
