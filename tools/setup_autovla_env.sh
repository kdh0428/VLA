#!/bin/bash
# AutoVLA PoC용 conda env `autovla` 구축.
# 원본 requirements.txt와의 차이:
#   - torch 2.4.0/cu121 은 RTX 5090(sm_120)을 지원하지 않아 torch 2.8.0+cu128 로 교체 (원 PoC 실행 버전과 동일)
#     (torchvision 0.23.0, triton은 torch가 가져오는 버전 사용)
#   - flash-attn / waymo / autoawq 는 설치하지 않음 (PoC는 eager attention, nuPlan만 사용)
set -euo pipefail

CONDA_DIR=/root/miniforge3
REQ=/root/VLA/autovla/requirements.txt

if [ ! -x $CONDA_DIR/bin/conda ]; then
    echo "[1/5] Miniforge 설치"
    wget -q -O /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/miniforge.sh -b -p $CONDA_DIR
    rm /tmp/miniforge.sh
    $CONDA_DIR/bin/conda init zsh bash >/dev/null
fi
source $CONDA_DIR/etc/profile.d/conda.sh

if ! conda env list | grep -q '^autovla '; then
    echo "[2/5] env 생성 (python 3.9, numpy 1.23.4)"
    conda create -y -n autovla python=3.9 "pip>=23.3" numpy=1.23.4
fi
conda activate autovla

echo "[3/5] torch 2.8.0 + cu128"
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

echo "[4/5] 나머지 requirements (torch/torchvision/triton 제외)"
grep -vE '^(torch|torchvision|triton)==' $REQ > /tmp/autovla_req.txt
pip install -r /tmp/autovla_req.txt
# requirements 중 일부가 torch를 다시 끌어내렸을 경우 대비
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install "numpy==1.23.4"

echo "[5/5] AutoVLA / navsim 패키지 설치 (editable, 의존성 재설치 없음)"
pip install --no-deps -e /root/VLA/autovla
[ -f /root/VLA/autovla/navsim/setup.py ] && pip install --no-deps -e /root/VLA/autovla/navsim || true

# 설치 후 발견된 호환성 수정
#   - libGL 없는 컨테이너: opencv-python -> headless (같은 버전)
#   - nuplan-devkit lidar.py가 pytest를 import, navsim이 몇몇 패키지를 import
#   - selenium이 urllib3 2.x를 끌어오지만 botocore 1.37은 <1.27 필요 (selenium은 PoC에서 미사용)
#   - torch가 GPU를 FASTEST_FIRST로 매겨 CUDA_VISIBLE_DEVICES=1이 3080 Ti가 됨 -> PCI 순서 고정
pip uninstall -y opencv-python
pip install opencv-python-headless==4.9.0.80 pytest nest_asyncio pyinstrument selenium ujson
pip install "urllib3<1.27"
conda env config vars set CUDA_DEVICE_ORDER=PCI_BUS_ID NUPLAN_MAPS_ROOT=/root/VLA/autovla/dataset/nuplan/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0
conda deactivate && conda activate autovla

python - <<'EOF'
import torch, transformers, numpy
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__, "numpy", numpy.__version__)
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
gi = 1 if torch.cuda.device_count() > 1 else 0
if "5090" not in torch.cuda.get_device_name(gi):
    print("WARNING: experiments were run on an RTX 5090 as GPU 1; on other GPUs set VLA_ANY_GPU=1 and expect small numeric differences")
x = torch.randn(1024, 1024, device=f"cuda:{gi}"); print("matmul ok", (x @ x).sum().item() != 0)
EOF
echo "ENV DONE"
