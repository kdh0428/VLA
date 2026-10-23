"""Scope/lock-condition check of the v04 TeX (task C). Prints every hit with context for manual review; writes JSON."""
import json, re, sys, os
P = sys.argv[1]
files = ["paper_body.tex", "supplement_body.tex", "supp_v03_svla.tex", "supp_v03_comparison.tex", "supp_v03_models.tex", "supp_v03_moved.tex", "llm_usage_statement_20261023_v04.tex"]
txt = {f: open(os.path.join(P, f)).read() for f in files if os.path.exists(os.path.join(P, f))}
pats = {"pre-registered (should be absent)": r"pre-?regist", "pre-specified": r"pre-?specif", "bitwise": r"bitwise",
        "uniqueness claims": r"\bonly\b[^.]{0,40}(model|polic|public)|\bunique\b|the only", "isolate/causal": r"isolat|causal",
        "improvement/success wording": r"improv|success rate|succeeds", "primary 0.048": r"0\.048", "drawer 0.145": r"0\.145",
        "drawer pose sensitivity": r"0\.075|0\.236|pose group|pose-group", "reset scenario": r"reset scenario", "D_S 12 components": r"D_\{?\\?(mathrm\{)?S|12 (normalized|motion)|twelve",
        "corresponding donor action": r"correspond", "ratio 0.8-1.25": r"0\.8\s*\\le|0\.8.{0,15}1\.25", "69/105 coverage": r"69/105|69\s*of\s*105|69/\s*105",
        "FAST autoregressive": r"FAST", "WorldVLA fork": r"fork|WorldVLA", "LLM statement": r"ChatGPT|Codex|Claude|LLM"}
out = {}
for name, pat in pats.items():
    hits = []
    for f, t in txt.items():
        for m in re.finditer(pat, t, re.I):
            a = max(0, m.start() - 110); b = min(len(t), m.end() + 110)
            hits.append({"file": f, "context": re.sub(r"\s+", " ", t[a:b])})
    out[name] = {"n": len(hits), "hits": hits}
    print(f"### {name}: {len(hits)}")
    for h in hits[:6]: print("  -", h["file"], "|", h["context"])
json.dump(out, open(sys.argv[2], "w"), indent=1)
