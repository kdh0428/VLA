# CVPR manuscript with visual evidence — 2026-10-19

The main PDF has **8 content pages + 1 references-only page**. The separate supplementary PDF has **8 pages**. The main paper contains **8 figures and 2 tables**, including original driving frames, matched plans, scene/drive heterogeneity, time-resolved responses, retained-window results, measured TCP paths, and success effects.

## Overleaf

1. Upload the entire source ZIP with its folders intact.
2. Set the main document to `main.tex` and compiler to **pdfLaTeX**.
3. Set the main document to `supplementary.tex` to build the supplementary PDF separately.

There is no root `output.pdf`, `fontspec`, or external-font dependency. Figures are in `figures/`; the bibliography is `references.bib`. Both documents compile independently.

## Local reproduction

```sh
python build.py
python verify_visual_evidence.py
```

Build dependencies: LaTeX/latexmk and Python with pypdf. The evidence verifier additionally needs NumPy and Pillow. It checks compact supplied plotting inputs; it does not rerun inference or coverage simulations.

To regenerate all scientific figures:

```sh
python make_figures.py
python make_design_figure.py
python make_visual_figures.py
python build.py
```

Figure regeneration needs NumPy, Matplotlib, Pillow, and pypdf. No GPU is used. `make_visual_figures.py` uses only CSV/JSON/JPEG inputs shipped here; PyArrow is not required. The selected TCP CSV was exported from the supplied Parquet traces and is recorded in the input hashes.

## Evidence and interval policy

- `evidence/CURRENT_CLAIM_NUMBERS.csv` is the current claim-number display table. `evidence/MANUSCRIPT_CLAIM_MAP.csv` maps claims to the revised manuscript.
- `evidence/latest_validation/` preserves the new interval policy, nine updated number rows, finite-frame/new-drive simulation results, analysis plan, estimand audit, and data-feasibility answer.
- `evidence/FINAL_PAPER_NUMBERS.*` and the earlier coverage files are frozen historical exports. Their old interval annotations are superseded by the current display policy; do not use them as the latest manuscript policy.
- `visual_inputs/` contains compact plotting data and all five supplied driving cases. `evidence/VISUAL_INPUT_HASHES.csv` records these exact bytes. `evidence/VISUAL_NUMERICAL_CHECKS.json` reports the implemented consistency checks.
- `evidence/VISUAL_UPDATE_PROVENANCE.json` identifies the original combined input ZIP by SHA-256. The original ZIP contains the full AutoVLA/SpatialVLA time-series Parquet files and raw extraction scripts; these large records are not duplicated in the paper source ZIP.
- `REVISION_NOTES.md` describes what changed and what remains unsupported. The original coding-agent requests are archived under `evidence/original_requests/`; they are completed requests, not instructions to repeat the work.

All new analysis is post hoc and uses stored outputs. Effect estimates and existing zero-null p values are unchanged. F's interaction interval is withdrawn. Eligible-frame intervals are explicitly approximate and have limited simulated support; selected intervals remain withheld. The interval audit does not establish day/vehicle independence.

AutoVLA frames show the stored current front view; the exact preprocessed multi-view tensors were not verified. AutoVLA trajectories are open-loop plans, not executed vehicle paths. SpatialVLA TCP paths are simulator measurements. Original robot RGB was not stored, so it is not presented as observed imagery. Example selection is outcome-conditioned and illustrative; pooled effects use the full eligible data.

## Format and remaining submission details

`cvpr.sty` and `ieeenat_fullname.bst` remain the unchanged **CVPR2026-v1(latex)** official files. Font size, margins, column separation, and text height are not reduced to fit. `build.py` requires exactly 8 main-content pages and checks the references boundary, official file hashes, undefined citations/references, and overfull boxes.

The target is CVPR 2027. Its linked AuthorGuidelines page was unavailable at the initial format check; this is a **provisional 2026-template review draft**, not a claim of verified 2027 compliance. Recheck the target-year author kit/policy and replace `paperID{*****}` in `preamble.tex` before submission. The visible placeholder is deliberate.

Official sources retained for provenance:
- https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines
- https://github.com/cvpr-org/author-kit
- https://cvpr.thecvf.com/Conferences/2027/CallForPapers
