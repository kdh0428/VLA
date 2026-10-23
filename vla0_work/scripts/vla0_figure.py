"""Timestep response figure for the VLA-0 pilot (reads analysis/timestep_curves_long.csv)."""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
A = sys.argv[1]
C = pd.read_csv(f"{A}/timestep_curves_long.csv")
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
PAL = {"O": "#2a78d6", "L": "#eb6834", "OL": "#1baf7a", "H": "#4a3aa7", "HL": "#e87ba4"}
LAB = {"O": "O: timestep 2 replaced", "L": "L: timestep 3 (latest) replaced", "OL": "OL: 2 and 3", "H": "H: 1 and 2", "HL": "HL: 1–3"}
fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.4))
for ax, suites, title in ((axs[0], None, "All suites"), (axs[1], "by suite", "L and O by suite")):
    if suites is None:
        for arm in ("O", "L", "OL", "H", "HL"):
            d = C[C.arm == arm]
            if d.empty:
                continue
            # observation means first, then mean over observations
            m = d.groupby(["obs_id", "timestep"]).resp.mean().groupby("timestep").mean()
            ax.plot(m.index, m.values, marker="o", ms=3, color=PAL[arm], ls="-" if arm in ("O", "L", "OL") else (0, (2, 1.5)), label=LAB[arm])
    else:
        for arm, ls in (("O", (0, (2, 1.5))), ("L", "-")):
            for s, col in zip(sorted(C.suite.unique()), ("#2a78d6", "#1baf7a", "#eda100")):
                d = C[(C.arm == arm) & (C.suite == s)]
                m = d.groupby("timestep").resp.mean()
                ax.plot(m.index, m.values, ls=ls, color=col, marker="o" if arm == "L" else "s", ms=2.5, label=f"{arm} {s.replace('libero_', '')}")
    ax.set_xlabel("generated timestep (after fixed emitted timesteps 1–3)"); ax.set_ylabel("mean |arm − R| / 1000 (dims 1–6)")
    ax.set_xticks([4, 5, 6, 7, 8]); ax.set_title(title, loc="left"); ax.grid(True, color="#e3e2de", lw=0.5)
    ax.legend(frameon=False, fontsize=5.2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.text(0.01, -0.06, "VLA-0 (LIBERO fine-tune), initial observations, development pilot. Response = normalised action-command change of the "
         "generated suffix relative to the self-patch reference R, not physical motion. Donors = length-matched vectors from 4 own samples (T = 0.7).",
         fontsize=5.4, color="#52514e")
fig.tight_layout()
for ext in ("pdf", "svg", "png"):
    fig.savefig(f"{A}/../figures/vla0_timestep_response.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
