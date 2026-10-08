#!/usr/bin/env python
"""
P1-C strata and S sampling from stage-1 output only (protocol §3). CPU.

  python p1c_select.py --stage1 <dir with records.jsonl/errors.jsonl> --logs <log> [<log> ...] --out <selection.json>

For each log segment: F = every eligible natural A- scene (pi = 1); S = the first min(4, N_S,seg) eligible natural A+
scenes in sha256("P1C-20261008|" + token) order (pi = m_seg / N_S,seg, w = 1 / pi). Scenes of the log without a stage-1
record are counted as `stage1_error` (with the error text) and are never selected. Refuses to overwrite --out.
Also writes <out>.tokens.json (selected tokens) and <out>.keep_files.json (image files the selected scenes need).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p1c_common as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--m", type=int, default=C.M_S)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    recs = {}
    for l in open(os.path.join(a.stage1, "records.jsonl")):
        if l.strip():
            r = json.loads(l); recs[r["token"]] = r
    errs = {}
    ep = os.path.join(a.stage1, "errors.jsonl")
    if os.path.exists(ep):
        for l in open(ep):
            if l.strip():
                e = json.loads(l); errs[e["token"]] = e["error"]
    rows = []
    for tok in C.tokens_of_logs(a.logs):
        r = recs.get(tok)
        if r is None:
            rows.append({"token": tok, "log": C.scene_index()[tok], "eligible": False, "a_eval": None, "t_star": None,
                         "exclusion_reason": "stage1_error" if tok in errs else "stage1_missing"})
            continue
        p = r["p1c"]
        rows.append({"token": tok, "log": r["log_name"], "eligible": p["eligible"], "a_eval": p["a_eval"],
                     "t_star": p["t_star"], "exclusion_reason": p["exclusion_reason"]})
    sel, den = C.select_strata(rows, m=a.m)
    for log, d in den.items():
        nerr = d["not_eligible_by_reason"].get("stage1_error", 0) + d["not_eligible_by_reason"].get("stage1_missing", 0)
        d["stage1_error_rate"] = nerr / max(1, d["scenes"])
    keep = set()
    for s in sel:
        keep.update(C.image_files_of_scene(json.load(open(os.path.join(C.SCENES, f"{s['token']}.json")))))
    out = {"logs": sorted(a.logs), "m": a.m, "salt": C.SALT, "t_star_max": C.T_STAR_MAX,
           "selected": sel, "denominators": den,
           "stage1_errors": {t: e for t, e in errs.items() if t in {r["token"] for r in rows}}}
    json.dump(out, open(a.out, "w"), indent=1)
    json.dump([s["token"] for s in sel], open(a.out + ".tokens.json", "w"))
    json.dump(sorted(keep), open(a.out + ".keep_files.json", "w"))
    nF = sum(s["stratum"] == "F" for s in sel); nS = len(sel) - nF
    print(f"[select] logs={len(a.logs)} scenes={len(rows)} eligible={sum(r['eligible'] for r in rows)} F={nF} S={nS} "
          f"keep_files={len(keep)}", flush=True)


if __name__ == "__main__":
    main()
