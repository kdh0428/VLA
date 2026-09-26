# PoC inference config: official CoT config + fp16 inference.
#
# WHY THIS FILE EXISTS
#   * orion_stage3_cot.py  gives the three reasoning answers we need for P/R labelling
#     (desc_qa=True), but runs in fp32 -- the 38.5 GB checkpoint does not fit in 32 GB.
#   * orion_stage3_fp16.py fits in memory but has desc_qa unset (=False), so the VQA
#     transform drops every reasoning round and only the planning round survives
#     (transforms_3d.py:1196) -- no text to label P or R with.
#
#   Neither upstream config is modified. This one inherits the CoT config wholesale and
#   overrides only what fp16 inference requires, matching orion_stage3_fp16.py exactly:
#     fp16_infer=True, fp16_eval=True on the model, and a top-level fp16=True so the
#     runner applies custom_wrap_fp16_model (det/map heads stay fp32).
#
#   fp16_eval=True is essential: with fp16_infer=True alone, simple_test_pts skips the
#   GT-dependent branch (orion.py:741) and never computes planner metrics or reads
#   ego_fut_trajs.  We need the GT trajectory, so eval stays on.
#
# Absolute paths are used so the runner works from any cwd.

_base_ = ["/root/VLA/orion/adzoo/orion/configs/orion_stage3_cot.py"]

fp16_infer = True

# NOTE the base config uses llm_path='ckpts/pretrain_qformer/' (cwd-relative) in BOTH the
# model and the test pipeline's tokenizer. Rather than duplicate the whole pipeline here
# just to absolutise one string, the runner chdir()s to /root/VLA/orion before building,
# exactly as the official ./adzoo/orion/orion_dist_eval.sh does. Data roots below are
# absolute anyway, so outputs do not depend on the cwd.

model = dict(
    fp16_infer=fp16_infer,
    fp16_eval=fp16_infer,
)

# Data roots as absolute paths (the base config uses cwd-relative ones).
data_root = "/root/VLA/orion/data/bench2drive"
info_root = "/root/VLA/orion/data/infos"

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,          # the runner drives the dataset directly, in clip order
    test=dict(
        data_root=data_root,
        ann_file=info_root + "/b2d_infos_val.pkl",
        map_root=data_root + "/maps",
        map_file=info_root + "/b2d_map_infos.pkl",
    ),
)

fp16 = True
