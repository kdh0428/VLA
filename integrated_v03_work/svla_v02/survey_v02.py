"""CPU layout survey (no model). Records full actor poses, robot base pose, instruction, and two fingerprints:
coarse (v01 rule: non-table actor xy rounded to 3 decimals) and full (all actor + robot poses rounded to 3 decimals + instruction).
  python survey_v02.py <task> <seed_from> <seed_to> <out.json>"""
import hashlib, json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import simpler_env
task, s0, s1, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
env = simpler_env.make(task); rows = []
for s in range(s0, s1 + 1):
    obs, info = env.reset(seed=s); u = env.unwrapped
    actors, coarse = {}, {}
    for a in u._scene.get_all_actors():
        n = a.name; p = a.pose
        actors[n] = {"p": [round(float(x), 4) for x in p.p], "q": [round(float(x), 4) for x in p.q]}
        if n in ("ground", "arena", "") or "table" in n.lower() or "background" in n.lower():
            continue
        coarse[n] = [round(float(x), 3) for x in list(p.p)[:2]]
    arts = {}
    for a in u._scene.get_all_articulations():
        p = a.pose; arts[a.name] = {"p": [round(float(x), 4) for x in p.p], "q": [round(float(x), 4) for x in p.q],
                                    "qpos": [round(float(x), 4) for x in a.get_qpos()]}
    instr = env.get_language_instruction()
    rp = arts.get(u.agent.robot.name, {})
    full_key = json.dumps({"actors": {k: [round(v, 3) for v in d["p"]] for k, d in actors.items()},
                           "arts_pose": {k: [round(v, 3) for v in d["p"]] + [round(v, 3) for v in d["q"]] for k, d in arts.items()},
                           "instr": instr}, sort_keys=True)
    rows.append({"seed": s, "instruction": instr, "coarse_fp": hashlib.sha256(json.dumps(sorted(coarse.items())).encode()).hexdigest()[:16],
                 "full_fp": hashlib.sha256(full_key.encode()).hexdigest()[:16], "actors": actors, "articulations": arts,
                 "info": {k: str(v) for k, v in (info or {}).items()}})
    print(task, s, rows[-1]["coarse_fp"], rows[-1]["full_fp"], instr, flush=True)
json.dump(rows, open(out, "w"), indent=1)
env.close()
