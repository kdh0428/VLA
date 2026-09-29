#!/bin/bash
# AutoVLA PoC 재현에 필요한 자산을 코드가 기대하는 경로로 내려받습니다.
#   - AutoVLA_PDMS_89.ckpt  -> /root/VLA/autovla_dl/AutoVLA/
#   - Qwen2.5-VL-3B-Instruct -> /root/VLA/autovla/Qwen2.5-VL-3B-Instruct/
#   - OpenScene navtest metadata + camera shard 0-5 (PoC 28개 log 전부 포함), nuPlan maps
#       -> /root/VLA/autovla/dataset/nuplan/{navsim_logs,sensor_blobs}/test
# wget -c 로 이어받기가 되므로 중단 후 다시 실행해도 됩니다.
set -euo pipefail

ROOT=/root/VLA
HF=https://huggingface.co
DL=$ROOT/autovla/dataset/nuplan/_download
AUTOVLA_COMMIT=ba34eed74ce6729e7986592d0e66cbaca397b4fa   # commit used for every experiment

echo "[0/5] AutoVLA repo"
if [ ! -d $ROOT/autovla/.git ]; then
    git clone https://github.com/ucla-mobility/AutoVLA.git $ROOT/autovla
    git -C $ROOT/autovla checkout $AUTOVLA_COMMIT
fi
mkdir -p $ROOT/autovla_dl/AutoVLA $ROOT/autovla/Qwen2.5-VL-3B-Instruct \
         $ROOT/autovla/dataset/nuplan/navsim_logs $ROOT/autovla/dataset/nuplan/sensor_blobs $DL

echo "[1/5] AutoVLA checkpoint"
wget -c -q --show-progress -O $ROOT/autovla_dl/AutoVLA/AutoVLA_PDMS_89.ckpt \
    $HF/Zewei-Zhou/AutoVLA/resolve/main/AutoVLA_PDMS_89.ckpt

echo "[2/5] Qwen2.5-VL-3B-Instruct"
for f in $(python3 -c "
import json,urllib.request
d=json.load(urllib.request.urlopen('$HF/api/models/Qwen/Qwen2.5-VL-3B-Instruct'))
print(' '.join(s['rfilename'] for s in d['siblings']))"); do
    wget -c -q -O $ROOT/autovla/Qwen2.5-VL-3B-Instruct/$f $HF/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main/$f
done

OS=$HF/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1
echo "[3/5] navtest metadata"
if [ ! -d $ROOT/autovla/dataset/nuplan/navsim_logs/test ]; then
    wget -c -q -O $DL/openscene_metadata_test.tgz $OS/openscene_metadata_test.tgz
    tar -xzf $DL/openscene_metadata_test.tgz -C $DL
    src=$DL/openscene-v1.1/meta_datas; [ -d $src/test ] && src=$src/test
    mv $src $ROOT/autovla/dataset/nuplan/navsim_logs/test
fi

echo "[4/5] camera shards 0-5"
for i in 0 1 2 3 4 5; do
    done_flag=$DL/camera_$i.done
    [ -f $done_flag ] && continue
    wget -c -q --show-progress -O $DL/camera_$i.tgz $OS/openscene_sensor_test_camera/openscene_sensor_test_camera_$i.tgz
    tar -xzf $DL/camera_$i.tgz -C $DL
    rm $DL/camera_$i.tgz
    touch $done_flag
done
mkdir -p $ROOT/autovla/dataset/nuplan/sensor_blobs/test
src=$DL/openscene-v1.1/sensor_blobs; [ -d $src/test ] && src=$src/test
mv $src/* $ROOT/autovla/dataset/nuplan/sensor_blobs/test/

echo "[5/5] nuPlan maps (navsim 장면 로드에 필요)"
if [ ! -d $ROOT/autovla/dataset/nuplan/maps ]; then
    wget -c -q -O $DL/maps.zip https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip
    python3 -c "import zipfile; zipfile.ZipFile('$DL/maps.zip').extractall('$ROOT/autovla/dataset/nuplan')"
    mv $ROOT/autovla/dataset/nuplan/nuplan-maps-v1.0 $ROOT/autovla/dataset/nuplan/maps
    rm $DL/maps.zip
fi
rm -rf $DL
echo "ALL DONE"
