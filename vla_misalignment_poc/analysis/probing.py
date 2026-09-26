"""
Layer-wise linear probing with clip-grouped splits.

Statistical rules enforced here (spec section 14):
  * splits are GROUPED BY CLIP -- frames of one clip never straddle train/test, so the
    strong temporal correlation between neighbouring frames cannot inflate scores;
  * StandardScaler is fit on TRAIN ONLY;
  * class imbalance is handled with `class_weight="balanced"`, and both AUROC and AUPRC
    are reported (AUPRC is the headline for rare labels such as pedestrian);
  * every probe is repeated over several seeds, and a bootstrap CI is available for the
    headline numbers;
  * a label-shuffled control is run alongside, giving the chance level for THIS split
    rather than an assumed 0.5.

Only linear models are used (logistic regression), as specified.
"""
from __future__ import annotations

import os

# Bound the BLAS thread pools BEFORE numpy/sklearn import them. These probes fit thousands
# of tiny logistic regressions on 4096-d vectors; letting each one fan out over every core
# oversubscribes the machine (load average >70 on 28 cores) and is slower than running them
# with a small fixed pool -- especially while the GPU extraction jobs are also running.
# Override with VLA_PROBE_THREADS if the box is idle.
_THREADS = os.environ.get("VLA_PROBE_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


#: L2 strength for every probe. The planning token is 4096-d while a group has at most
#: ~1.8k samples, so an under-regularised probe interpolates the training folds (train
#: accuracy 1.000 at C=1.0) and generalises worse. Measured on lead_vehicle @ L24:
#:   C=1.0   AUROC 0.764, shuffled 0.566, 16.1 s
#:   C=0.01  AUROC 0.802, shuffled 0.498,  7.2 s
#:   C=0.001 AUROC 0.821, shuffled 0.476,  4.0 s   <- better AND ~4x faster
#: Strong regularisation also pulls the shuffled null back towards 0.5, which is what a
#: null should look like.
PROBE_C = 1e-3

#: A probe whose positives live in fewer clips than this cannot be shown to generalise
#: ACROSS clips -- with clip-grouped folds the classifier can only ever separate a handful
#: of scenes, so a high AUROC may just be clip identity. Reported, not silently dropped.
MIN_POS_CLIPS = 5
#: ... likewise if a few clips supply most of the positives.
MAX_TOP3_CLIP_SHARE = 0.55


@dataclass
class ProbeResult:
    layer: int
    label: str
    n: int
    n_pos: int
    auroc: float = float("nan")
    auprc: float = float("nan")
    balanced_acc: float = float("nan")
    macro_f1: float = float("nan")
    accuracy: float = float("nan")
    # chance level measured on the same splits with shuffled labels (mean over
    # `n_shuffles` permutations x seeds; sd exposes how noisy the null is)
    auroc_shuffled: float = float("nan")
    auprc_shuffled: float = float("nan")
    auroc_shuffled_sd: float = float("nan")
    baserate: float = float("nan")
    auroc_ci: tuple[float, float] = (float("nan"), float("nan"))
    auprc_ci: tuple[float, float] = (float("nan"), float("nan"))
    per_seed: list[dict] = field(default_factory=list)
    n_folds: int = 0
    # --- clip-level support, needed to judge whether the score means anything ---
    n_clips: int = 0
    n_clips_pos: int = 0
    n_clips_neg: int = 0
    top3_clip_share_pos: float = float("nan")
    reliable: bool = False
    reliability_note: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["auroc_ci"] = list(self.auroc_ci)
        d["auprc_ci"] = list(self.auprc_ci)
        return d


def _clip_support(y: np.ndarray, groups: np.ndarray) -> dict:
    """How many distinct clips carry the positive / negative class, and how concentrated."""
    gp = groups[y == 1]
    gn = groups[y == 0]
    uniq, cnt = np.unique(gp, return_counts=True)
    top3 = float(np.sort(cnt)[::-1][:3].sum() / max(cnt.sum(), 1)) if cnt.size else float("nan")
    return {"n_clips": int(len(np.unique(groups))),
            "n_clips_pos": int(len(uniq)),
            "n_clips_neg": int(len(np.unique(gn))),
            "top3_clip_share_pos": top3}


def _bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0,
                  alpha: float = 0.05) -> tuple[float, float]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    if v.size == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def probe_binary(X: np.ndarray, y: np.ndarray, groups: np.ndarray, layer: int, label: str,
                 seeds: tuple[int, ...] = (0, 1, 2), n_splits: int = 5,
                 C: float = PROBE_C, max_iter: int = 1000,
                 run_shuffle_control: bool = True, n_shuffles: int = 2) -> ProbeResult:
    """
    Cross-validated linear probe for one binary label at one layer.

    Uses StratifiedGroupKFold so each fold keeps the class ratio while never splitting a
    clip. Out-of-fold scores are pooled before computing the metrics, which is more stable
    than averaging per-fold AUCs on small folds.

    The result carries clip-level support (`n_clips_pos`, `top3_clip_share_pos`) and a
    `reliable` flag. When the positives sit in only a handful of clips, a clip-grouped
    probe cannot demonstrate generalisation ACROSS scenes and the AUROC may be reporting
    clip identity -- the number is still returned, but flagged.
    """
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    res = ProbeResult(layer=layer, label=label, n=len(y), n_pos=int(y.sum()))
    res.baserate = float(y.mean()) if len(y) else float("nan")
    for k, v in _clip_support(y, groups).items():
        setattr(res, k, v)

    notes = []
    if res.n_clips_pos < MIN_POS_CLIPS:
        notes.append(f"positives in only {res.n_clips_pos} clips (<{MIN_POS_CLIPS})")
    if np.isfinite(res.top3_clip_share_pos) and res.top3_clip_share_pos > MAX_TOP3_CLIP_SHARE:
        notes.append(f"top-3 clips hold {res.top3_clip_share_pos:.0%} of positives")
    res.reliability_note = "; ".join(notes)
    res.reliable = not notes

    n_groups = len(np.unique(groups))
    if res.n_pos < 8 or (len(y) - res.n_pos) < 8 or n_groups < 3:
        return res                                   # not enough signal to probe honestly

    k = min(n_splits, n_groups, res.n_pos, len(y) - res.n_pos)
    if k < 2:
        return res
    res.n_folds = k

    aurocs, auprcs, baccs, f1s, accs = [], [], [], [], []
    sh_auroc, sh_auprc = [], []

    for seed in seeds:
        try:
            sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            splits = [(tr, te) for tr, te in sgkf.split(X, y, groups=groups)]
            oof = np.full(len(y), np.nan)

            # Cache the per-fold scaling once; the shuffle control reuses the same folds
            # and the same scaler, so it differs from the real probe only in the labels.
            prepared = []
            for tr, te in splits:
                sc = StandardScaler().fit(X[tr])           # fit on TRAIN ONLY
                Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
                prepared.append((tr, te, Xtr, Xte))
                if len(np.unique(y[tr])) < 2:
                    continue
                clf = LogisticRegression(max_iter=max_iter, C=C, class_weight="balanced")
                clf.fit(Xtr, y[tr])
                oof[te] = clf.predict_proba(Xte)[:, 1]

            m = np.isfinite(oof)
            if m.sum() < 10 or len(np.unique(y[m])) < 2:
                continue
            yt, ps = y[m], oof[m]
            aurocs.append(roc_auc_score(yt, ps))
            auprcs.append(average_precision_score(yt, ps))
            pred = (ps >= 0.5).astype(int)
            baccs.append(balanced_accuracy_score(yt, pred))
            f1s.append(f1_score(yt, pred, average="macro", zero_division=0))
            accs.append(float((pred == yt).mean()))
            res.per_seed.append({"seed": seed, "auroc": aurocs[-1], "auprc": auprcs[-1],
                                 "balanced_acc": baccs[-1]})

            # Null distribution: several independent permutations per seed. One permutation
            # is far too noisy under clip-grouped folds -- with positives concentrated in a
            # few clips a single shuffle can land anywhere between ~0.2 and ~0.9.
            if run_shuffle_control:
                for j in range(n_shuffles):
                    rng = np.random.default_rng(seed * 1000 + j)
                    y_sh = rng.permutation(y)
                    oof_sh = np.full(len(y), np.nan)
                    for tr, te, Xtr, Xte in prepared:
                        if len(np.unique(y_sh[tr])) < 2:
                            continue
                        c2 = LogisticRegression(max_iter=max_iter, C=C,
                                                class_weight="balanced")
                        c2.fit(Xtr, y_sh[tr])
                        oof_sh[te] = c2.predict_proba(Xte)[:, 1]
                    ms = np.isfinite(oof_sh)
                    if ms.sum() >= 10 and len(np.unique(y[ms])) >= 2:
                        sh_auroc.append(roc_auc_score(y[ms], oof_sh[ms]))
                        sh_auprc.append(average_precision_score(y[ms], oof_sh[ms]))
        except Exception:
            continue

    if aurocs:
        res.auroc = float(np.mean(aurocs))
        res.auprc = float(np.mean(auprcs))
        res.balanced_acc = float(np.mean(baccs))
        res.macro_f1 = float(np.mean(f1s))
        res.accuracy = float(np.mean(accs))
        res.auroc_ci = _bootstrap_ci(aurocs)
        res.auprc_ci = _bootstrap_ci(auprcs)
    if sh_auroc:
        res.auroc_shuffled = float(np.mean(sh_auroc))
        res.auprc_shuffled = float(np.mean(sh_auprc))
        res.auroc_shuffled_sd = float(np.std(sh_auroc))
    return res


@dataclass
class MultiClassResult:
    layer: int
    label: str
    n: int
    classes: list[str]
    accuracy: float = float("nan")
    balanced_acc: float = float("nan")
    macro_f1: float = float("nan")
    accuracy_shuffled: float = float("nan")
    macro_f1_shuffled: float = float("nan")
    majority_baseline: float = float("nan")
    per_seed: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def probe_multiclass(X: np.ndarray, y: np.ndarray, groups: np.ndarray, layer: int,
                     label: str, seeds: tuple[int, ...] = (0, 1, 2), n_splits: int = 5,
                     C: float = PROBE_C, max_iter: int = 1000) -> MultiClassResult:
    """Multinomial linear probe (used for coarse action semantics)."""
    y = np.asarray(y)
    classes = sorted(set(map(str, y)))
    res = MultiClassResult(layer=layer, label=label, n=len(y), classes=classes)
    if len(classes) < 2 or len(y) < 30:
        return res
    counts = np.array([np.sum(y == c) for c in classes])
    res.majority_baseline = float(counts.max() / counts.sum())
    if counts.min() < 5 or len(np.unique(groups)) < 3:
        return res

    k = min(n_splits, len(np.unique(groups)), int(counts.min()))
    if k < 2:
        return res

    accs, baccs, f1s, sh_accs, sh_f1s = [], [], [], [], []
    for seed in seeds:
        try:
            sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            oof = np.empty(len(y), dtype=object)
            oof_sh = np.empty(len(y), dtype=object)
            rng = np.random.default_rng(seed)
            y_sh = rng.permutation(y)
            for tr, te in sgkf.split(X, y, groups=groups):
                if len(np.unique(y[tr])) < 2:
                    continue
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(max_iter=max_iter, C=C, class_weight="balanced")
                clf.fit(sc.transform(X[tr]), y[tr])
                oof[te] = clf.predict(sc.transform(X[te]))
                if len(np.unique(y_sh[tr])) >= 2:
                    c2 = LogisticRegression(max_iter=max_iter, C=C, class_weight="balanced")
                    c2.fit(sc.transform(X[tr]), y_sh[tr])
                    oof_sh[te] = c2.predict(sc.transform(X[te]))
            m = np.array([v is not None for v in oof])
            if m.sum() < 10:
                continue
            yt, pr = y[m], np.array(list(oof[m]))
            accs.append(float((pr == yt).mean()))
            baccs.append(balanced_accuracy_score(yt, pr))
            f1s.append(f1_score(yt, pr, average="macro", zero_division=0))
            res.per_seed.append({"seed": seed, "acc": accs[-1], "macro_f1": f1s[-1]})
            ms = np.array([v is not None for v in oof_sh])
            if ms.sum() >= 10:
                ys, prs = y[ms], np.array(list(oof_sh[ms]))
                sh_accs.append(float((prs == ys).mean()))
                sh_f1s.append(f1_score(ys, prs, average="macro", zero_division=0))
        except Exception:
            continue

    if accs:
        res.accuracy = float(np.mean(accs))
        res.balanced_acc = float(np.mean(baccs))
        res.macro_f1 = float(np.mean(f1s))
    if sh_accs:
        res.accuracy_shuffled = float(np.mean(sh_accs))
        res.macro_f1_shuffled = float(np.mean(sh_f1s))
    return res


def load_hidden(path: str, layer: int, pool: str = "mean") -> np.ndarray | None:
    """
    Load one frame's planning-token block for one layer as a 1-D fp32 vector.

    A frame can contain several `<waypoint_ego>` positions; ORION's planner consumes the
    whole (n, 4096) block, so we pool to a single vector for probing. Mean pooling is the
    default; 'first'/'last' are available for ablation.
    """
    try:
        with np.load(path) as z:
            key = str(layer)
            if key not in z.files:
                return None
            a = z[key].astype(np.float32)
    except Exception:
        return None
    if a.ndim == 1:
        return a
    if a.shape[0] == 0:
        return None
    if pool == "first":
        return a[0]
    if pool == "last":
        return a[-1]
    return a.mean(axis=0)


class HiddenCache:
    """
    Read every frame's .npz ONCE and keep all layers in memory.

    Without this, each (layer, group) combination re-opens all ~1.8k npz files -- with 33
    layers and several groups that is >200k file opens and dominates the probe runtime.
    The whole cache is small: 1809 frames x 33 layers x 4096 fp16 = ~0.5 GB.
    """

    def __init__(self, pool: str = "mean"):
        self.pool = pool
        self._by_id: dict[str, dict[int, np.ndarray]] = {}
        self.layers: list[int] = []

    def load(self, rows: list[dict]) -> "HiddenCache":
        for r in rows:
            sid, hp = r.get("sample_id"), r.get("hidden_path")
            if not hp or sid in self._by_id:
                continue
            try:
                with np.load(hp) as z:
                    d: dict[int, np.ndarray] = {}
                    for k in z.files:
                        a = z[k].astype(np.float32)
                        if a.ndim > 1:
                            if a.shape[0] == 0:
                                continue
                            a = (a[0] if self.pool == "first" else
                                 a[-1] if self.pool == "last" else a.mean(axis=0))
                        d[int(k)] = a
                if d:
                    self._by_id[sid] = d
            except Exception:
                continue
        if self._by_id:
            self.layers = sorted(next(iter(self._by_id.values())).keys())
        return self

    def get(self, sample_id: str, layer: int) -> np.ndarray | None:
        return self._by_id.get(sample_id, {}).get(layer)


def build_matrix(rows: list[dict], layer: int, pool: str = "mean",
                 cache: "HiddenCache | None" = None) -> tuple[np.ndarray, list[int]]:
    """Stack per-frame vectors for `layer`; returns (X, kept_row_indices)."""
    vecs, keep = [], []
    for i, r in enumerate(rows):
        if cache is not None:
            v = cache.get(r.get("sample_id"), layer)
        else:
            hp = r.get("hidden_path")
            v = load_hidden(hp, layer, pool) if hp else None
        if v is None:
            continue
        vecs.append(v)
        keep.append(i)
    if not vecs:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.stack(vecs).astype(np.float32), keep
