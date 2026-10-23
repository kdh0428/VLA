"""Build v03 (main + supplementary) in an isolated temp directory with tectonic, verify, then atomically export the PDFs.
  python build_v03.py [path/to/tectonic]
Checks: no LaTeX errors, no undefined references/citations, no overfull boxes, main references start on page 9 (8 content
pages), unchanged official template hashes. Writes BUILD_QA_20261023_v03.json next to this script."""
import hashlib, json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path
P = Path(__file__).resolve().parent
TEC = sys.argv[1] if len(sys.argv) > 1 else "tectonic"
qa = {"engine": "tectonic 0.15.0 (XeTeX-based) in an isolated temp dir; v02 was built with pdfLaTeX, and the unmodified v02 sources give the same page boundaries under tectonic (main refs p.9, supplement refs p.14)", "documents": {}}
with tempfile.TemporaryDirectory(prefix="cvpr-v03-") as d:
    B = Path(d)
    for f in P.iterdir():
        if f.suffix in {".tex", ".bib", ".bst", ".sty"}: shutil.copy2(f, B / f.name)
    shutil.copytree(P / "figures", B / "figures")
    for n in ("main", "supplementary"):
        r = subprocess.run([TEC, "-X", "compile", f"{n}.tex", "--keep-logs", "--keep-intermediates"], cwd=B, capture_output=True, text=True, timeout=900)
        (B / f"{n}_build_console_20261023_v03.txt").write_text(r.stdout + r.stderr)
        if r.returncode: raise SystemExit(r.stdout[-3000:] + r.stderr[-3000:])
        log = (B / f"{n}.log").read_text(errors="replace"); aux = (B / f"{n}.aux").read_text()
        m = re.search(r"\\newlabel\{page:references\}\{\{[^}]*\}\{(\d+)\}", aux)
        data = (B / f"{n}.pdf").read_bytes(); assert data.rstrip().endswith(b"%%EOF")
        x = {"first_references_page": int(m.group(1)) if m else None, "latex_errors": bool(re.search(r"^!|LaTeX Error|Fatal error", log, re.M)),
             "undefined_refs_or_citations": bool(re.search(r"Citation .+ undefined|Reference .+ undefined|There were undefined", log)),
             "overfull_boxes": log.count("Overfull"), "sha256": hashlib.sha256(data).hexdigest()}
        assert not x["latex_errors"] and not x["undefined_refs_or_citations"] and not x["overfull_boxes"], (n, x)
        if n == "main": assert x["first_references_page"] == 9, x
        qa["documents"][n] = x
    for n in ("main", "supplementary"):
        for suf in (".pdf", ".log", ".aux", "_build_console_20261023_v03.txt"):
            src = B / f"{n}{suf}"; tgt = P / f"{n}{suf}"; tmp = tgt.with_name(tgt.name + ".tmp")
            tmp.write_bytes(src.read_bytes()); os.replace(tmp, tgt)
exp = {"cvpr.sty": "2602473285d1a7df2a445ac89b76e1afa0acab78e056f0369d19770245190153", "ieeenat_fullname.bst": "e38e6166bd7b1e6d23a1b79dcdb55c656e4fcdbe91bdf6b50d827e6b5d1aacfc"}
qa["template_hashes_unchanged"] = all(hashlib.sha256((P / k).read_bytes()).hexdigest() == v for k, v in exp.items()); assert qa["template_hashes_unchanged"]
(P / "BUILD_QA_20261023_v03.json").write_text(json.dumps(qa, indent=1))
print(json.dumps(qa, indent=1))
