#!/bin/bash
REV=6c62a511ad56042b2b3ea0a90c4def33e1ea3b96
D=/dev/shm/vla0/model_last
for f in model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do
  for i in $(seq 1 30); do
    curl -sS -L --http1.1 -C - --retry 5 -o "$D/$f" "https://huggingface.co/ankgoyal/vla0-libero/resolve/$REV/model_last/$f" && break
    echo "retry $f $i"; sleep 3
  done
done
sha256sum $D/*.safetensors
echo DONE
