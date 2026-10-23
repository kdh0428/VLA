# source this: SimplerEnv (ManiSkill2_real2sim, SAPIEN 2.2.2) on the autovla conda env, software Vulkan (Mesa lavapipe), RTX 5090 for the policy
source /root/miniforge3/etc/profile.d/conda.sh && conda activate autovla
export PYTHONPATH=/root/VLA/spatialvla/site:/root/VLA/simpler/site:/root/VLA/simpler/SimplerEnv:/root/VLA/simpler/SimplerEnv/ManiSkill2_real2sim:/root/VLA/autovla_misalignment_poc/scripts/cross_domain_temporal
export LD_LIBRARY_PATH=/root/VLA/openvla/mesa/lib:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=/root/VLA/openvla/mesa/share/vulkan/icd.d/lvp_icd.x86_64.json
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false
