"""CPU: reset SimplerEnv tasks for candidate seeds and fingerprint the physical layout (object names + poses after reset).
No model. Output: manifest/layout_survey.json"""
import hashlib, json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import simpler_env
out = {}
plan = {"google_robot_move_near": list(range(0, 60)),
        "google_robot_pick_coke_can": list(range(0, 3)) + list(range(2000, 2030)) + list(range(3000, 3012))}
for task, seeds in plan.items():
    env = simpler_env.make(task); rows = []
    for s in seeds:
        obs, info = env.reset(seed=s)
        u = env.unwrapped
        objs = {}
        for a in u._scene.get_all_actors():
            n = a.name
            if n in ("ground", "arena", "") or "table" in n.lower() or "background" in n.lower():
                continue
            p = a.pose; objs[n] = [round(float(x), 3) for x in list(p.p)[:2]]
        instr = env.get_language_instruction()
        fp = hashlib.sha256(json.dumps(sorted(objs.items())).encode()).hexdigest()[:16]
        extra = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in (info or {}).items() if k in ("episode_id", "model_id", "obj_init_options", "episode_model_ids")}
        rows.append({"seed": s, "instruction": instr, "objects_xy": objs, "layout_fp": fp, "info": extra})
        print(task, s, fp, instr, flush=True)
    out[task] = rows
    env.close()
json.dump(out, open("/root/VLA/svla_wc_work/manifest/layout_survey.json", "w"), indent=1)
