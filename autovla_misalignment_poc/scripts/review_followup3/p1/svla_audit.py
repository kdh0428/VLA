#!/usr/bin/env python
"""
SpatialVLA P1-B / P1-C audit analysis (CPU only). PROTOCOL_P1.md §8.4.
usage:
  svla_audit.py pilot RUN1 RUN2 OUT_DIR       -> svla_tolerances.json (pilot: natural vs natrep, natural vs separate process)
  svla_audit.py gate  RUN1 RUN2 TOL OUT_PREFIX -> STATE_RESTORE_AUDIT.csv/.md + svla_noop_rows.csv

Comparison of an arm with the root N of the same (task, seed, cfg), step by step over the steps both cover
(root-like arms: t = 0..79; branched arms: the replayed prefix is checked through `restore`, then t >= T0):
  executed float64 action (bytes and max |diff|), every fingerprint component (hash equality), TCP max |diff| (m),
  executor fingerprint. First divergent step and the components that differ there are recorded.
Tolerance rule (fixed before the pilot): if the pilot shows bitwise reruns (all actions bytes-equal and every component
hash equal), T = 0 (bitwise) for every no-op arm. Otherwise T_action / T_tcp = 2 x the pilot maximum and hashes are
reported, not required.
Pass: U0 == U0' == U0'' == U1 == U2 over all steps; U3 == U0 for t >= T0 and restore ok; F/C/R restore ok (identical
full-state fingerprint and executor state at every replayed step up to T0).
"""
import csv
import json
import sys
from collections import defaultdict

import numpy as np

COMP = ("sim_state", "agent_state", "qpos", "qvel", "qacc", "drive_target", "ctrl_arm", "ctrl_gripper", "grip", "steps",
        "seeds", "rng", "timestep", "cameras", "img_raw", "img224")


def load(run):
    eps = defaultdict(dict)
    for line in open(f"{run}/episodes.jsonl"):
        e = json.loads(line)
        eps[(e["task"], e["seed"], e["cfg"])][e["mode"]] = e
    return eps


def cmp(root, arm, t_from=0):
    ra = {r["t"]: r for r in root["rec"]}
    rb = {r["t"]: r for r in arm["rec"]}
    ts = sorted(t for t in set(ra) & set(rb) if t >= t_from)
    first, comps, amax, tmax, ex_eq, act_eq = None, [], 0.0, 0.0, True, True
    for t in ts:
        x, y = ra[t], rb[t]
        a1, a2 = np.asarray(x["action"]), np.asarray(y["action"])
        amax = max(amax, float(np.abs(a1 - a2).max()))
        tmax = max(tmax, float(np.abs(np.asarray(x["fp"]["tcp"]) - np.asarray(y["fp"]["tcp"])).max()))
        bad = [c for c in COMP if x["fp"].get(c) != y["fp"].get(c)]
        if x["action_hex"] != y["action_hex"]:
            act_eq = False; bad = ["action"] + bad
        if x["exfp"] != y["exfp"]:
            ex_eq = False; bad.append("executor")
        if bad and first is None:
            first, comps = t, bad
    gen_equal = all((ra[t].get("exec"), ra[t].get("ctx")) == (rb[t].get("exec"), rb[t].get("ctx")) for t in ts)
    return {"n_steps": len(ts), "exec_ctx_tokens_equal_all_generations": gen_equal, "steps_equal_len": len(ra) == len(rb) if t_from == 0 else None,
            "actions_bytes_equal": act_eq, "action_maxabs": amax, "tcp_maxabs_m": tmax, "executor_equal": ex_eq,
            "first_divergent_step": first, "components_at_first": comps, "success_equal": root["success"] == arm["success"]}


def a5split(run, csv_path, md_path):
    """AMENDMENT P1-02: null arm vs natural with the split executor fingerprint. PASS iff actions bytes-equal, every
    simulator component equal and exfp_policy equal at every one of the 80 steps; exfp_cf reported only."""
    eps = load(run)
    out = []
    for k, d in sorted(eps.items()):
        if "natural" not in d or "nullfull" not in d:
            continue
        ra = {r["t"]: r for r in d["natural"]["rec"]}; rb = {r["t"]: r for r in d["nullfull"]["rec"]}
        ts = sorted(set(ra) & set(rb))
        act = all(ra[t]["action_hex"] == rb[t]["action_hex"] for t in ts)
        sim_bad = sorted({c for t in ts for c in COMP if ra[t]["fp"].get(c) != rb[t]["fp"].get(c)})
        pol = all(ra[t]["exfp_policy"] == rb[t]["exfp_policy"] for t in ts)
        cf_diff = [t for t in ts if ra[t]["exfp_cf"] != rb[t]["exfp_cf"]]
        tcp = max(float(np.abs(np.asarray(ra[t]["fp"]["tcp"]) - np.asarray(rb[t]["fp"]["tcp"])).max()) for t in ts)
        ok = act and not sim_bad and pol and len(ts) == 80 and len(ra) == len(rb)
        out.append({"control": "C2_A5_split", "task": k[0], "seed": k[1], "cfg": k[2], "arm": "nullfull", "pass": ok,
                    "n_steps": len(ts), "actions_bytes_equal": act, "sim_components_differing": sim_bad,
                    "exfp_policy_equal_all_steps": pol, "tcp_maxabs_m": tcp,
                    "exfp_cf_differs_steps_first": cf_diff[:1], "exfp_cf_n_steps_differ": len(cf_diff),
                    "run": run})
    old = list(csv.DictReader(open(csv_path))) if __import__("os").path.exists(csv_path) else []   # old rows kept verbatim
    oldkeys = list(old[0]) if old else []
    keys = oldkeys + [k for k in out[0] if k not in oldkeys]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in old:
            w.writerow(r)
        for r in out:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    n_ok = sum(r["pass"] for r in out)
    with open(md_path, "a") as f:
        f.write("\n## C2 re-verification with the split executor fingerprint (AMENDMENT_P1_02, CP2 decision A5)\n\n")
        f.write(f"Run `svla_gate/a5_rerun/` (natural + nullfull, same 20 units, 4 workers, RTX 5090). Pass rule fixed before the "
                f"run: executed actions bytes-equal + every simulator fingerprint component equal + policy-state hash equal at all "
                f"80 steps (tolerance 0); shadow-cf bookkeeping hash reported only.\n\n")
        f.write(f"| control | pass / n |\n|---|---|\n| C2_A5_split | {n_ok}/{len(out)} |\n\n")
        f.write(f"- actions bytes-equal: {sum(r['actions_bytes_equal'] for r in out)}/{len(out)}; simulator components all equal: "
                f"{sum(not r['sim_components_differing'] for r in out)}/{len(out)}; policy-state hash equal at every step: "
                f"{sum(r['exfp_policy_equal_all_steps'] for r in out)}/{len(out)}; max TCP diff {max(r['tcp_maxabs_m'] for r in out)} m.\n")
        f.write(f"- bookkeeping (reported only): shadow-cf hash differs in {sum(r['exfp_cf_n_steps_differ'] > 0 for r in out)}/{len(out)} "
                f"units, first at t = {sorted({r['exfp_cf_differs_steps_first'][0] for r in out if r['exfp_cf_differs_steps_first']})}.\n")
        f.write("- Earlier rows are kept unchanged above: strict pre-fixed rule 0/20, A5 post-hoc policy-state check 20/20.\n")
        fails = [r for r in out if not r["pass"]]
        f.write(f"- Failures: {len(fails)}" + "".join(f"\n  - {r['task']} {r['seed']} {r['cfg']}: {r}" for r in fails) + "\n")
    print("C2_A5_split", n_ok, "/", len(out))


def main():
    mode = sys.argv[1]
    if mode == "a5split":
        a5split(*sys.argv[2:5])
        return
    if mode == "pilot":
        r1, r2, od = sys.argv[2:5]
        e1, e2 = load(r1), load(r2)
        rows = []
        for k, d in e1.items():
            if "natural" in d and "natrep" in d:
                rows.append({"cmp": "c5a", "key": k, **cmp(d["natural"], d["natrep"])})
            if "natural" in d and k in e2 and "natural" in e2[k]:
                rows.append({"cmp": "c5b", "key": k, **cmp(d["natural"], e2[k]["natural"])})
        bitwise = all(r["first_divergent_step"] is None for r in rows) and len(rows) > 0
        tol = {"bitwise": bitwise, "T_action": 0.0 if bitwise else 2 * max(r["action_maxabs"] for r in rows),
               "T_tcp_m": 0.0 if bitwise else 2 * max(r["tcp_maxabs_m"] for r in rows), "n_pairs": len(rows),
               "pairs": [{k_: (list(v) if k_ == "key" else v) for k_, v in r.items()} for r in rows], "rule": __doc__}
        json.dump(tol, open(f"{od}/svla_tolerances.json", "w"), indent=1)
        print({k: v for k, v in tol.items() if k not in ("pairs", "rule")})
        return
    r1, r2, tolp, outp = sys.argv[2:6]
    tol = json.load(open(tolp))
    e1, e2 = load(r1), load(r2)
    c4 = [json.loads(l) for l in open(f"{r1}/c4_t0.jsonl")]
    out = []

    def judge(r, need_full):
        if tol["bitwise"]:
            ok = r["first_divergent_step"] is None
        else:
            ok = r["action_maxabs"] <= tol["T_action"] and r["tcp_maxabs_m"] <= tol["T_tcp_m"]
        if need_full and r.get("steps_equal_len") is False:
            ok = False
        return ok
    for k, d in sorted(e1.items()):
        root = d.get("natural")
        if root is None:
            continue
        for arm, ctrl, t_from in (("natrep", "C5a", 0), ("hooknone", "C1", 0), ("nullfull", "C2", 0), ("branchN", "C3_restore", 8)):
            if arm in d:
                r = cmp(root, d[arm], t_from)
                ok = judge(r, t_from == 0)
                strict = ok
                if arm == "nullfull" and not ok:     # AMENDMENT A5 (post hoc, reported next to the strict verdict)
                    only_cf = r["components_at_first"] == ["executor"] and r["actions_bytes_equal"] and \
                        r["exec_ctx_tokens_equal_all_generations"] and r["tcp_maxabs_m"] == 0.0
                    r["strict_fail_reason"] = "executor fingerprint differs (includes P1-D shadow cf dict filled by null mode)"
                    r["pass_policy_state_A5"] = bool(only_cf)
                    ok = bool(only_cf)
                r["pass_strict_prefixed_rule"] = strict
                if arm == "branchN":
                    ok = ok and bool(d[arm]["restore"]["ok"])
                out.append({"control": ctrl, "task": k[0], "seed": k[1], "cfg": k[2], "arm": arm, "pass": ok,
                            "restore_ok": (d[arm]["restore"] or {}).get("ok"), **r})
        if k in e2 and "natural" in e2[k]:
            r = cmp(root, e2[k]["natural"])
            out.append({"control": "C5b", "task": k[0], "seed": k[1], "cfg": k[2], "arm": "natural(run2)", "pass": judge(r, True),
                        "restore_ok": None, **r})
        for arm in ("feedback", "corrected", "reverse"):
            if arm in d:
                rs = d[arm]["restore"]
                out.append({"control": "P1C_branch_restore", "task": k[0], "seed": k[1], "cfg": k[2], "arm": arm,
                            "pass": bool(rs["ok"]), "restore_ok": rs["ok"], "n_steps": rs["n_steps"],
                            "components_at_first": [c for x in rs["component_diffs"] for c in x["components"]][:6],
                            "executor_equal": rs["executor_fp_equal"]})
    for x in c4:
        out.append({"control": "C4_t0_report", "task": x["task"], "seed": x["seed"], "cfg": x["cfg"], "arm": "T0 chunk",
                    "pass": None, "b1_equal_batch8": x["b1_equal_batch8"], "generate_equal_batch8": x["generate_equal_batch8"]})
    keys = sorted({k for r in out for k in r})
    with open(f"{outp}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in out:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    summ = defaultdict(lambda: [0, 0])
    for r in out:
        if r["pass"] is not None:
            summ[r["control"]][0] += int(r["pass"]); summ[r["control"]][1] += 1
    c4s = [r for r in out if r["control"] == "C4_t0_report"]
    with open(f"{outp}.md", "w") as f:
        f.write("# STATE_RESTORE_AUDIT — SpatialVLA P1-B / P1-C (PROTOCOL_P1.md §8)\n\n")
        f.write(f"Tolerance (pilot): bitwise = {tol['bitwise']}, T_action = {tol['T_action']}, T_tcp = {tol['T_tcp_m']} m.\n\n")
        f.write("| control | pass / n |\n|---|---|\n" + "".join(f"| {c} | {a}/{n} |\n" for c, (a, n) in sorted(summ.items())))
        strict = [r for r in out if r.get("pass_strict_prefixed_rule") is False]
        f.write(f"\nStrict pre-fixed rule (executor fingerprint incl. the shadow cf dict): {len(strict)} arm comparisons fail "
                f"({sorted({r['control'] for r in strict})}); see AMENDMENT_P1_01.md A5.\n")
        f.write(f"\nC4 analogue at T0 (report only): B=1 decode == batch-8 chunk {sum(r['b1_equal_batch8'] for r in c4s)}/{len(c4s)}; "
                f"HF generate == batch-8 chunk {sum(r['generate_equal_batch8'] for r in c4s)}/{len(c4s)}.\n\n")
        fails = [r for r in out if r["pass"] is False]
        f.write(f"Failures: {len(fails)}\n\n" + "".join(f"- {r['control']} {r['task']} seed {r['seed']} {r['cfg']} {r['arm']}: "
                                                       f"first divergent step {r.get('first_divergent_step')}, components {r.get('components_at_first')}, "
                                                       f"action max {r.get('action_maxabs')}, TCP max {r.get('tcp_maxabs_m')}\n" for r in fails))
    print(dict(summ), "fails", sum(1 for r in out if r["pass"] is False))


if __name__ == "__main__":
    main()
