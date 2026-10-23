"""Independent CPU verifier (second implementation; no GPU, no simulator). From raw token ids in records.jsonl only:
re-decodes actions with the official action tokenizer, re-derives donor eligibility, E_L/E_O/Delta per observation,
layout -> task -> two-task means, matched subset, and denominators; compares with analysis/summary.json.
  python verify_cpu.py run/pilot/records.jsonl analysis/summary.json"""
import json, sys
from collections import defaultdict
import numpy as np
from transformers import AutoProcessor
tok = AutoProcessor.from_pretrained("/root/VLA/spatialvla/spatialvla-4b-224-sft-fractal", trust_remote_code=True).action_tokenizer
rng3 = [(tok.translation_tokenizer.token_start_idx, tok.translation_tokenizer.token_end_idx),
        (tok.rotation_tokenizer.token_start_idx, tok.rotation_tokenizer.token_end_idx),
        (tok.gripper_tokenizer.token_start_idx, tok.gripper_tokenizer.token_end_idx)]
ok_t = lambda ids: all(rng3[i % 3][0] <= v <= rng3[i % 3][1] for i, v in enumerate(ids)) and 1 not in ids  # noqa: E731
dec = lambda ids: tok.decode_token_ids_to_actions(np.array(ids, dtype=np.int64).reshape(-1, 3))[:, :6]  # noqa: E731
recs = [json.loads(l) for l in open(sys.argv[1])]; S = json.load(open(sys.argv[2]))
obs = []
cnt = defaultdict(int)
for r in recs:
    cnt["records"] += 1
    n12 = r["official"]["ids"][:12]
    if not (len(n12) == 12 and ok_t(n12)): continue
    cnt["natural_valid"] += 1
    if r["R"]["ids"][:6] != n12[6:]: continue
    cnt["R_identity"] += 1
    uN = dec(n12); donor = None
    for s in r.get("sources", []):
        a = s["ids"][:6]
        if len(a) == 6 and ok_t(a) and a[2] == n12[2] and a[5] == n12[5]:
            ua = dec(a + n12[6:])
            dO, dL = np.abs(ua[0] - uN[0]).mean(), np.abs(ua[1] - uN[1]).mean()
            if dO > 0 and dL > 0: donor = (a, dO, dL); break
    if donor is None: continue
    cnt["donor_found"] += 1
    a, dO, dL = donor
    arm = {}
    for k, cond in (("O", a[:3] + n12[3:6]), ("L", n12[:3] + a[3:6])):
        g = [x for x in r["arms"][k]["ids"][:6]]
        assert r["arms"][k]["cond"] == cond
        if not ok_t(g): break
        arm[k] = dec(n12[:6] + g)[2:]
    if len(arm) < 2: continue
    uR = dec(n12[:6] + r["R"]["ids"][:6])[2:]
    EL, EO = np.abs(arm["L"] - uR).mean(), np.abs(arm["O"] - uR).mean()
    obs.append((r["task"], r["seed"], EL - EO, 0.8 <= dL / dO <= 1.25))
cnt["analyzed"] = len(obs); cnt["matched"] = sum(o[3] for o in obs)
def two(rows):
    lay = defaultdict(list)
    for t, s, v, _ in rows: lay[(t, s)].append(v)
    tm = defaultdict(list)
    for (t, s), v in lay.items(): tm[t].append(np.mean(v))
    return (np.mean([np.mean(v) for v in tm.values()]) if len(tm) == 2 else None), {t: len(v) for t, v in tm.items()}
pa, pl = two([o for o in obs if o[3]]); aa, al = two(obs)
chk = {"flow": [cnt["records"] == S["flow"]["records"], cnt["natural_valid"] == S["flow"]["natural_valid"], cnt["R_identity"] == S["flow"]["R_identity"],
                cnt["donor_found"] == S["flow"]["donor_found"], cnt["analyzed"] == S["flow"]["analyzed_joint_valid"], cnt["matched"] == S["flow"]["matched"]],
       "primary_Delta": abs(pa - S["primary_matched"]["Delta"]["two_task"]) < 1e-12,
       "primary_layouts": pl == S["primary_matched"]["Delta"]["n_layouts"],
       "all_Delta": abs(aa - S["all_joint_valid"]["Delta"]["two_task"]) < 1e-12, "all_layouts": al == S["all_joint_valid"]["Delta"]["n_layouts"]}
ok = all(chk["flow"]) and all(v for k, v in chk.items() if k != "flow")
print(json.dumps({"counts": cnt, "primary_two_task_Delta": pa, "all_two_task_Delta": aa, "checks": chk, "ALL_MATCH": ok}, indent=1, default=float))
