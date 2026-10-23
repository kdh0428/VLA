"""Shared paths/helpers for the CVPR visual-evidence package (post-hoc, CPU only, read-only inputs)."""
import hashlib
import os

POC = "/root/VLA/autovla_misalignment_poc"
REPO = "/root/VLA"
OUT = f"{POC}/outputs/cvpr_visual_evidence_20261019"
B2 = f"{POC}/outputs/review_followup4/B/B2"
P1C = f"{POC}/outputs/review_followup/P1C_new_log_replication/run_20261008"
P1A = f"{POC}/outputs/review_followup/P1A_gt_free_reference/run_20261008"
R8R9 = f"{POC}/outputs/review_followup/R8R9_raw_recovery/run_20261008"
P1D = f"{POC}/outputs/review_followup/P1D_dose_control/run_20261008"
NUPLAN = f"{REPO}/autovla/dataset/nuplan"
BNUM = f"{POC}/outputs/review_followup5/B_NUMERIC"
P2 = f"{POC}/outputs/review_followup3/P2"
CVPR_NEXT = f"{POC}/outputs/cvpr_next_validation_20261019"
AQ = f"{POC}/outputs/manuscript_author_queries_20261019"

# reference categorical palette (dataviz skill default, light mode), fixed order, never cycled
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"
GT_COLOR = "#0b0b0b"


def sha(p, n=None):
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return h if n is None else h[:n]


def rel(p):
    return os.path.relpath(p, REPO)


def mpl_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
                         "grid.color": GRID, "grid.linewidth": 0.5, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "font.family": "DejaVu Sans", "lines.linewidth": 1.4, "figure.dpi": 150,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def save(fig, name):
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{OUT}/figures/{name}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
