#!/usr/bin/env python
"""CPU-only: count model/generation calls from stored per-observation/per-call records (no inference)."""
import json, glob, collections
out = {}
def svla(path, tag):
    c = collections.Counter(); peak = None
    for l in open(path):
        r = json.loads(l); c["observations"] += 1
        c["official_natural_generate"] += 1 if "official" in r else 0
        c["adapter_natural_forced"] += 1 if "adapter_natural" in r else 0
        c["R_selfpatch_forced"] += 1 if "R" in r else 0
        c["source_sampling_generate"] += len(r.get("sources", []))
        c["arm_forced"] += len(r.get("arms", {}) or {})
        c["status_" + str(r.get("status"))] += 1
        c["t_s_sum"] += r.get("t_s", 0) or 0
    c["model_calls_total"] = c["official_natural_generate"] + c["adapter_natural_forced"] + c["R_selfpatch_forced"] + c["source_sampling_generate"] + c["arm_forced"]
    out[tag] = dict(c)
svla("/root/VLA/svla_wc_work/run/pilot/records.jsonl", "svla_discovery_pilot_attempt2")
svla("/root/VLA/svla_wc_work/run/feasibility/records.jsonl", "svla_discovery_validation_attempt3")
svla("/root/VLA/integrated_v03_work/svla_v02/run/main/records.jsonl", "svla_followup_v02_main")
svla("/root/VLA/integrated_v03_work/svla_v02/run/smoke/records.jsonl", "svla_followup_v02_smoke")
svla("/root/VLA/integrated_v03_work/svla_drawer/run/recovered/records.jsonl", "svla_drawer_recovered")
svla("/root/VLA/integrated_v03_work/svla_drawer/run/main_attempt1_collection_interrupted/records.jsonl", "svla_drawer_main_attempt1")
# VLA-0 pilot
c = collections.Counter(); pk = 0
for l in open("/root/VLA/vla0_work/pkg/run/pilot/records.jsonl"):
    r = json.loads(l); c["observations"] += 1; c["generate_calls"] += r.get("n_generate_calls", 0)
    c["latency_s_sum"] += r.get("latency_s", 0) or 0; pk = max(pk, r.get("peak_vram_gb", 0) or 0)
    c["source_returns"] += len(r.get("sources", []))
c["peak_vram_gb_max"] = pk; out["vla0_pilot"] = dict(c)
# VLA-0 closed loop
c = collections.Counter()
for f in glob.glob("/root/VLA/vla0_cl_work/run/attempt1/*/*/calls.jsonl"):
    for l in open(f):
        r = json.loads(l); c["policy_calls"] += 1; c["status_" + str(r.get("status"))] += 1
        for k in ("t_natural_s", "t_sources_s", "t_forced_s"):
            if r.get(k) is not None: c["gen_" + k[2:-2]] += 1
        c["t_call_s_sum"] += r.get("t_call_s", 0) or 0
c["generation_calls_total"] = c["gen_natural"] + c["gen_sources"] + c["gen_forced"]
out["vla0_closed_loop"] = dict(c)
json.dump(out, open("/root/VLA/v05_work/A_materials/A2_cost_call_counts_20261023.json", "w"), indent=1)
print(json.dumps(out, indent=1))
