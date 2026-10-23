# Paper revision changelog — 20261023 v04

Input: `CVPR_INTEGRATED_REVISION_20261023_v03.zip`, SHA256 `4ee9d73ce52de9a43eb8e7bebb27b1b2547bbfbcc1d1cea2ac0849b0791857b1`.

- Retained the original experiment records, protocols, source samples, snapshots, and supplied CPU verifiers unchanged.
- Defined SpatialVLA D_S explicitly over 12 normalized motion components; clarified position-specific corresponding donor actions and the fixed intermediate action.
- Replaced publication claims of registration with pre-specification, without rewriting historical protocols.
- Distinguished drawer reset scenarios from physical pose groups; added explicitly post-hoc seven-group bootstrap sensitivities with the original scenario weights and alternative equal-group weights.
- Narrowed model-selection claims to the inspected implementations, corrected the FAST autoregressive characterization and checkpoint-size wording, and attributed the inspected WorldVLA fork.
- Removed internal experiment-version labels from regenerated SpatialVLA publication figures. Updated units in captions and figure labels; plotted values remain unchanged.
- Restored two VLA-0 raw archives and the SpatialVLA discovery raw archive from earlier supplied packages. Added portable raw-token/pose verification and publication-figure reproduction.
- Retrieved the official CVPR 2027 v1 author kit through the current Author Guidelines page, replaced the previous 2026 style with the unmodified 2027 style, and recorded the source hashes and diff. The 2027 style removes visible paper IDs; no margins or font sizes were manually changed.
- Added the mandatory LLM Usage Statement between the main text and references. It describes known ChatGPT/Codex use; the authors must confirm completeness across all tools used.
- Built both PDFs from clean temporary directories using pdfLaTeX and BibTeX with official CVPR 2027 style hashes. Main content is eight pages; the usage statement and references start on page nine. Supplement has 18 content pages and one reference page.
- Current QA: 25,957 new-data checks and 668 baseline checks pass. These are CPU consistency checks, not fresh GPU inference or simulator runs.

The initial v04 textual edit ledger is retained as `REVIEW_CHANGES_20261023_v04.json`; the final source-file differences and added files are recorded in `qa/FINAL_CHANGE_AUDIT_20261023_v04.json`.
