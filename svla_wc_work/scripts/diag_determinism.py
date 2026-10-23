"""GPU diagnostic for run-to-run reproducibility (2026-10-22). Repeats official generate, prompt-prefill logits, adapter
natural and R self-patch on the pilot observations whose official/collection/adapter/R tokens disagreed, plus 3 controls.
  python diag_determinism.py --mode default|deterministic --out FILE   (deterministic: CUBLAS_WORKSPACE_CONFIG set by caller)"""
import argparse, json, os, sys, time, hashlib
import numpy as np, torch, cv2 as cv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wc_run import Ledger  # noqa: E402
from svla_wc_core import SVLA  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
W = "/root/VLA/svla_wc_work"
led = Ledger(os.path.join(W, "run", "gpu_ledger.json"), 150.0); led.tick(f"diag_{a.mode}")
if a.mode == "deterministic":
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "set CUBLAS_WORKSPACE_CONFIG"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
m = SVLA()
bad = json.load(open(os.path.join(W, "manifest", "diag_obs_ids.json")))
coll = {o["obs_id"]: o for o in json.load(open(os.path.join(W, "run", "pilot", "collected.json"))) if "obs_id" in o}
ctrl = [k for k in coll if k not in bad][:3]
res = []
for oid in bad + ctrl:
    o = coll[oid]
    img = cv.imread(os.path.join(W, "run", "pilot", o["png"]))[:, :, ::-1].copy()
    assert hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest() == o["img_sha256"]
    inp = m.inputs(img, o["instruction"])
    offs = [m.official(inp)[0][:13] for _ in range(4)]
    pre = []
    for _ in range(3):
        c, lg, P = m._prefill(inp); pre.append(lg[0].cpu()); del c
    nat = [m.forced(inp, [], 13)[0] for _ in range(2)]
    a12 = offs[0][:12]
    Rs = [m.forced(inp, a12[:6], 7)[0][:6] for _ in range(2)]
    res.append({"obs_id": oid, "control": oid in ctrl, "n_distinct_official": len({tuple(x) for x in offs}),
                "official_eq_collect": [x == o["collect_predict_action_ids"][:13] for x in offs],
                "prefill_logit_maxdiff_across_repeats": float(max((pre[i] - pre[0]).abs().max() for i in range(1, 3))),
                "n_distinct_adapter_natural": len({tuple(x) for x in nat}), "adapter_eq_official0": [x == offs[0] for x in nat],
                "R_identity_vs_official0": [r == a12[6:12] for r in Rs]})
    led.tick()
    print(res[-1], flush=True)
json.dump({"mode": a.mode, "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "results": res}, open(a.out, "w"), indent=1)
led.tick()
