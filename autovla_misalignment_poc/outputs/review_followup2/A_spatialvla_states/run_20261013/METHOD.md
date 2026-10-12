# A2 (SpatialVLA): physical initial-state clusters — method (fixed before clustering)

Written 2026-10-12/13 before `cluster_states.py` was run. This is a post-hoc audit of existing data
(experiments 34, 35, P1-D); nothing here is pre-registered. No model, no GPU (CPU SAPIEN + Mesa lavapipe).
Code: `scripts/review_followup2/svla_reset_states.py` (re-creates resets, writes raw states),
`scripts/review_followup2/cluster_states.py` (tolerance, clustering, manifest, counts). Existing files are not modified.

## 1. What is re-created

| experiment | task | seeds / units | reset call used by the runner (verified in code) |
|---|---|---|---|
| 34 (`cross_domain_temporal/svla_rollouts.py`) | coke, move_near | 0–39 | `env.reset(seed=seed)` (no options, env object reused inside a long-lived worker) |
| 35 (`spatialvla_protection/svla_protection.py`) | coke, move_near | 0–39 (`rollouts`), 40–119 (`rollouts_ext`) | `env.reset(seed=seed)` (same as 34) |
| P1-D (`review_followup/p1d/p1d_runner.py`, main run) | coke | seeds 200–319 | `env.reset(seed, options={"reconfigure": True})` |
| P1-D | move_near | 120 units of `units.json` | `env.reset(seed, options={"reconfigure": True, "obj_init_options": {"episode_id": eid}})` |

Envs: `google_robot_pick_coke_can` = `GraspSingleOpenedCokeCanInScene-v0`, `google_robot_move_near` = `MoveNearGoogleBakedTexInScene-v1`
(simpler_env.make, visual-matching prepackaged config).

Canonical re-creation = a reset with `reconfigure=True` (scene rebuilt), which P1-D reported to be bit-reproducible.
Experiments 34/35 did *not* reconfigure; their exact float state depended on which episode ran before in the same
worker (P1-D protocol §0.2 reports ~2e-6 differences). That history is not recorded, so their exact bitwise state is
**not recoverable**; we (a) assign them to the canonical state of their seed, and (b) measure the size of the
history-dependent drift (default resets after other seeds) to check that it is far below the tolerance. If the drift
ever exceeds the tolerance, the affected rows are marked `physical state uncertain` (the tolerance is not changed).

## 2. What is recorded per reset (right after `reset()`, before any step)

- discrete specification: task, language instruction, source/target object names (move_near), coke orientation label,
  object model ids and scales, robot `urdf_version`, `rgb_overlay_path` (background inpainting image), scene name,
  episode_id (move_near; recorded as a label only, not used as a clustering key).
- continuous state: every dynamic actor (sorted by name): position p (m), quaternion q (sign-canonicalised so the
  first nonzero of (w,x,y,z) is positive), linear velocity (m/s), angular velocity (rad/s); robot root pose,
  joint qpos (rad / m), joint qvel. Also the raw `env.unwrapped.get_state()` vector and its exact sha1 (float64,
  same `H()` as P1-D) and the sha1 of the 224×224 observation image (same resize as the runners).

Note on "overlay": in P1-D files `overlay` = robot `urdf_version` (recolored robot visual meshes). The URDF diff shows
only visual-mesh paths change; collision meshes, inertias and joints are identical. The background `rgb_overlay_path`
is fixed per task.

## 3. Cluster definitions (three nested levels)

All levels: discrete keys must match exactly, and the continuous state must match within the tolerance (§4).
RNG seed, experiment, condition, execution config (r1e4/r4e1/...), `reconfigure` flag and episode_id label are
**never** keys.

- **initial_state_cluster_id (primary)** = task + goal specification (instruction, source/target object) + asset ids
  (object model ids/scales, robot urdf_version, rgb_overlay_path, scene) + continuous state.
  This is the "same evaluation problem, same observation" unit.
- **scene_state_cluster_id** = as primary but without the goal specification (instruction/source/target).
  Same scene and same rendered image except nothing — the instruction is the only difference.
- **physical_state_cluster_id** = only physics-relevant content: task family, object model ids/scales, continuous
  state (drops goal and the visual-only robot urdf_version). Coarsest unit: identical physics at t=0.

The primary level is what A2 asks for; the coarser levels are reported because two primary clusters that share a
physical layout are not independent draws of the initial-state distribution.

## 4. Float tolerance (fixed here, not changed after clustering)

Element-wise absolute tolerance, applied per component type:

| component | tolerance | justification |
|---|---|---|
| positions (m), quaternion components, joint qpos | **1e-4** | SAPIEN stores poses/qpos in float32: ulp ≤ 2.4e-7 for \|x\| ≤ 2. Largest known non-physical discrepancy is the history-dependent reset drift of ~2e-6 (P1-D §0.2). 1e-4 (0.1 mm) is ≥ 400× ulp and ≥ 40× that drift, and ≥ 1000× below the designed differences of the benchmark (move_near layouts differ by ≥ 0.15 m; coke xy is continuous). Exp 35 `pose0` is stored at float32 precision, P1-D hashes exact float64 copies of float32 values. |
| linear velocities (m/s) | **1e-3** | the env's own "at rest" criterion (`_initialize_actors`: extra settling if Σ\|v\| > 1e-3) |
| angular velocities, joint qvel (rad/s) | **1e-2** | same criterion (Σ\|ω\| > 1e-2) |

Clustering = connected components (single linkage) of the "all components within tolerance" relation among
states with identical discrete keys. Diagnostics reported but not used to change anything: largest within-cluster
deviation and smallest between-cluster deviation (in units of tolerance), and any component whose diameter exceeds
the tolerance (chaining).

## 5. Restore verification vs. complete_restore_hash

`initial_state_cluster_id` answers "is this the same physical starting problem?". `complete_restore_hash` would answer
"can the run be resumed bit-exactly?" and must include simulator internals (get_state: actor poses/velocities,
articulation qpos/qvel), controller targets and gripper state (not part of get_state), policy-side buffers
(SpatialVLA chunk/ensemble buffer, sticky-gripper state), and RNG state (decode is greedy, so no policy RNG; reset RNG
is `RandomState(seed)`). What each experiment verified is reported per row (`restore_verification`):

- P1-D: recorded `init.state_hash` (exact sha1 of get_state float64) and `img_hash` for every episode, plus per-step
  replay checks (`restore.ok`) and executor deep-copy. We check our re-created hashes against the recorded ones.
- Exp 34: no state hash recorded. Saved t=0 frames (`frames/<task>_s<seed>/000.png`, last condition run) are compared
  pixel-wise with the re-created observation.
- Exp 35: no state hash; only TCP `pose0` (robot only, identical for all seeds) and the instruction are stored; these
  are compared. Physical object state of exp 35 is therefore inferred from the seed → reset mapping (the same code path
  as exp 34), not verified from run records.

## 6. Determinism checks

(i) canonical reset repeated in the same process after other seeds, (ii) canonical reset in a fresh process,
(iii) default (no-reconfigure) reset after other seeds: drift magnitude vs canonical, both after random predecessors
and in a sequential 0..119 chain.
