# PAPER_REVISION_CHANGELOG_20261023_v03 (base: `CVPR_PAPER_REVISION_20261021_v02.zip`, author TeX)

## Main text (`paper/paper_body.tex`)
- **Abstract.** The last sentences now report the SpatialVLA follow-up: 0.048 [0.026, 0.072]; 69 observations, 34 layouts, two tasks. They say "observed in a second backbone family, without isolating a backbone effect" and drop the v02 "one backbone family" sentence.
- **Contributions.** Item 1 adds the SpatialVLA fresh-cache self-patch path. Item 2 becomes "three policies", noting that the intermediate action is held fixed.
- **New §3.4 "SpatialVLA: within-call action edits".** Defines the edit, the intermediate-action condition and the response E_k. §3.5 adds E_L − E_O as the outcome.
- **Table 1.** Three rows added: SpatialVLA discovery, follow-up, open drawer (supplementary).
- **New §5.4 "SpatialVLA: the latest-action pattern in a second backbone family".** Covers the fresh-cache path, the discovery result as exploratory, the pre-registered follow-up, coverage, balanced donor distance, change rates, D(OL, L) and the drawer result. New Table 5 lists the separate experiments without pooling them.
- **Discussion.** The three policies' estimands are distinguished (Supp. S).
- **Limitations.** States the backbone scope (observed, not isolated) and summarizes model selection (Supp. T).
- **Conclusion.** Updated.
- **Moved to the new Supp. U** to keep 8 pages. The numbers stay in the main text, and references point to Supp. Figs. S12–S14:
  - AutoVLA time-resolved figure;
  - VLA-0 closed-loop trace figure;
  - new SpatialVLA layout-level figure.
- **Layout fix.** The SpatialVLA cross-step figure is defined earlier in the source, so it no longer floats onto page 9.
- **Unchanged.** Every v02 number and statement: the v02 raw verifier passes 668/668 on the v03 TeX.

## Supplement (`paper/supplement_body.tex` + new inputs)
Sections A–O are unchanged. New sections:
- **P.** SpatialVLA path: decoding facts, adapter, the cache-reuse finding and fix, and the discovery pilot.
- **Q.** Matched follow-up: design, denominators (Table S13), results (Table S14), component and donor-distance figures (Fig. S10).
- **R.** Open-drawer supplementary task: selection, the interrupted collection and amendment, results (Fig. S11).
- **S.** Cross-policy intervention comparison (Table S15).
- **T.** Model selection (Table S16).
- **U.** Figures moved from the main text (Figs. S12–S14).

Supplement: 18 content pages + references (was 13 + references).

## Corrections made before export
- The a4 E_L value in Table S14 was first typed as 0.073 and corrected to **0.072** (exact 0.07247), after the automated number check.

## Build
- tectonic 0.15.0 (XeTeX-based), because pdfLaTeX is not installed on this server. The unmodified v02 sources reproduce v02's page boundaries under tectonic.
- Main: references on p. 9 (8 content pages), 10 pages total. Supplement: references on p. 19, 19 pages total.
- 0 LaTeX errors, 0 undefined references or citations, 0 overfull boxes. Official template hashes are unchanged.
- PDFs are exported atomically from an isolated build directory (`paper/build_v03.py`, `paper/BUILD_QA_20261023_v03.json`).
- **Recheck needed:** the authors' pdfLaTeX build must be rechecked for the 8-page boundary.
