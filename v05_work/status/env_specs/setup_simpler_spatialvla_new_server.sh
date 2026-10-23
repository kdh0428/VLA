#!/bin/bash
# New-server rebuild of /root/VLA/simpler and /root/VLA/spatialvla (not runnable from git: libsapien.so is 209 MB > GitHub limit).
# Versions/commits recorded on the old server 2026-10-23 (target_site_dists_simpler_spatialvla_20261023.txt, simplerenv_commit_20261023.txt).
# Python = conda env autovla (Python 3.9). Run step by step and check each result.
set -euo pipefail
PY=/root/miniforge3/envs/autovla/bin/python

# 1) SimplerEnv + ManiSkill2_real2sim at the recorded commits (old server had no local modifications)
mkdir -p /root/VLA/simpler
cd /root/VLA/simpler
[ -d SimplerEnv ] || git clone https://github.com/simpler-env/SimplerEnv.git
cd /root/VLA/simpler/SimplerEnv
git checkout eddb5690749a1e34267807c0f7b5eede323918d7
git submodule update --init --recursive
git -C ManiSkill2_real2sim checkout ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682

# 2) target-installed packages used through PYTHONPATH (see /root/VLA/simpler/env.sh)
"$PY" -m pip install --no-deps --target /root/VLA/simpler/site \
  sapien==2.2.2 gymnasium==0.29.1 h5py==3.11.0 mediapy==1.2.7 ruckig==0.19.4 transforms3d==0.4.2 \
  trimesh==4.12.2 gitpython==3.2.0 gitdb==4.0.12 smmap==5.0.3 farama-notifications==0.0.6
"$PY" -m pip install --no-deps --target /root/VLA/spatialvla/site transformers==4.47.0 tokenizers==0.21.0

# 3) SpatialVLA checkpoint; compare sha256 with spatialvla_ckpt_fingerprint_from_v02_main.json
"$PY" -m huggingface_hub.commands.huggingface_cli download IPEC-COMMUNITY/spatialvla-4b-224-sft-fractal \
  --local-dir /root/VLA/spatialvla/spatialvla-4b-224-sft-fractal

# 4) Renderer: the old server used Mesa lavapipe from /root/VLA/openvla/mesa (software Vulkan; see env.sh).
#    If that directory is not copied, switch env.sh to the NVIDIA Vulkan ICD and record the renderer change
#    as a protocol amendment (images can differ, so do not compare with old-server rollouts).
echo "done; now: source /root/VLA/simpler/env.sh && python -c 'import simpler_env; simpler_env.make(\"google_robot_pick_coke_can\")'"
