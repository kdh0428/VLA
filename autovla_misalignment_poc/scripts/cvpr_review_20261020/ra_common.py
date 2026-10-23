"""Shared paths / helpers (CVPR review response 2026-10-20; post_hoc/exploratory; CPU only)."""
import hashlib
import os
import sys

POC = "/root/VLA/autovla_misalignment_poc"
REPO = "/root/VLA"
OUT = f"{POC}/outputs/cvpr_review_and_contribution_20261020"
SCR = f"{POC}/scripts"
B2 = f"{POC}/outputs/review_followup4/B/B2"
P1C = f"{POC}/outputs/review_followup/P1C_new_log_replication/run_20261008"
P1A = f"{POC}/outputs/review_followup/P1A_gt_free_reference/run_20261008"
P1D = f"{POC}/outputs/review_followup/P1D_dose_control/run_20261008"
DEV_AHC = f"{POC}/outputs/action_history_causal"
RF5 = f"{POC}/outputs/review_followup5"
AQ = f"{POC}/outputs/manuscript_author_queries_20261019"
CN = f"{POC}/outputs/cvpr_next_validation_20261019"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

for p in (SCR, f"{SCR}/review_followup3", f"{SCR}/review_followup2", f"{SCR}/review_followup/p1ab"):
    if p not in sys.path:
        sys.path.insert(0, p)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rel(p):
    return os.path.relpath(p, REPO)


def drive_of(log):
    s = log.split("_"); return f"{s[0]}_{s[1]}"


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "legend.fontsize": 6.3, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": INK2, "grid.color": GRID, "grid.linewidth": 0.5, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "lines.linewidth": 1.4, "font.family": "DejaVu Sans"})
    return plt


def save(fig, name, sub="figures"):
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{OUT}/{sub}/{name}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
