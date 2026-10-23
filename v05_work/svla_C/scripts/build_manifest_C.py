"""Build the frozen C manifest from CPU surveys (no model output used). Rules in PROTOCOL_C_SVLA_FULL_EPISODE_20261023_v01.md."""
import json, collections
M = "/root/VLA/v05_work/svla_C/manifest/"
def rpose(r):
    k = [a for a in r["articulations"] if a.startswith("google")][0]
    return json.dumps(r["articulations"][k]["p"] + r["articulations"][k]["q"])
units = []
# coke: first 20 seeds in order with unique coarse fp (all disjoint from earlier surveys); cluster = layout
coke = json.load(open(M + "survey_google_robot_pick_coke_can_6000_6039.json")); seen = set()
for r in coke:
    if r["coarse_fp"] in seen: continue
    seen.add(r["coarse_fp"]); units.append({"task": "google_robot_pick_coke_can", "seed": r["seed"], "instruction": r["instruction"],
        "layout_id": "coke_" + r["coarse_fp"], "cluster_id": "coke_" + r["coarse_fp"], "full_fp": r["full_fp"], "reuse": "new seed range 6000+; coarse fp disjoint from all earlier coke surveys"})
    if sum(u["task"].endswith("coke_can") for u in units) == 20: break
# move_near: only 10 physical layouts exist (coarse fp); 2 states per layout = first two seeds with distinct instructions
mn = json.load(open(M + "survey_google_robot_move_near_6000_6079.json")); per = collections.OrderedDict()
for r in mn:
    lst = per.setdefault(r["coarse_fp"], [])
    if len(lst) < 2 and all(x["instruction"] != r["instruction"] for x in lst): lst.append(r)
for fp, lst in per.items():
    for r in lst:
        units.append({"task": "google_robot_move_near", "seed": r["seed"], "instruction": r["instruction"], "episode_id": r["info"].get("episode_id"),
            "layout_id": "mn_" + fp, "cluster_id": "mn_" + fp, "full_fp": r["full_fp"], "reuse": "physical layout reused (registry has 10 move_near layouts, all used in earlier studies); instruction/state new"})
# close drawer (third task): first 20 seeds with unique (robot pose, instruction); cluster = robot pose group
cd = json.load(open(M + "survey_google_robot_close_drawer_6000_6059.json")); seen = set(); groups = {}
for r in cd:
    key = (rpose(r), r["instruction"])
    if key in seen: continue
    seen.add(key); g = groups.setdefault(rpose(r), f"cd_pg{len(groups)}")
    units.append({"task": "google_robot_close_drawer", "seed": r["seed"], "instruction": r["instruction"], "layout_id": f"{g}_{r['instruction'].split()[1]}",
        "cluster_id": g, "full_fp": r["full_fp"], "reuse": "new task (close drawer not used before); robot pose groups shared with earlier open-drawer surveys"})
    if sum(u["task"].endswith("close_drawer") for u in units) == 20: break
for i, u in enumerate(units):
    short = {"google_robot_pick_coke_can": "coke", "google_robot_move_near": "mn", "google_robot_close_drawer": "cd"}[u["task"]]
    u["unit_id"] = f"C_{short}_s{u['seed']}"
val = {}
for u in units:
    if val.setdefault(u["task"], []) .__len__() < 2: val[u["task"]].append(u["unit_id"])
summary = {t: {"n_states": sum(u["task"] == t for u in units), "n_layouts": len({u["layout_id"] for u in units if u["task"] == t}),
               "n_clusters": len({u["cluster_id"] for u in units if u["task"] == t})} for t in val}
json.dump({"units": units, "self_validation_units": val, "summary": summary, "anchor_call": 8,
           "seed_ranges_surveyed": {"coke": "6000-6039", "move_near": "6000-6079", "close_drawer": "6000-6059", "open_drawer(not used)": "6000-6059"}},
          open(M + "C_manifest_20261023_v01.json", "w"), indent=1)
print(json.dumps(summary, indent=1), val)
