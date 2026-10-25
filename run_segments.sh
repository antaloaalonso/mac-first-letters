#!/bin/bash
# usage: run_segments.sh <segments_dir> <volume_url> <out_name> [render args...]
# Renders every tifxyz under <segments_dir>, runs ink_9um both directions, writes JPG previews.
set -u
SEGS=$1; VOL=$2; NAME=$3; shift 3
cd "$(dirname "$0")"; source .venv/bin/activate
export PYTHONPATH=$PWD/villa/vesuvius/src PYTORCH_ENABLE_MPS_FALLBACK=1
CK=${CK:-checkpoints/ink_9um/hybrid_3d2d-seed42/step-075000.pth}
ENS=${ENS---ensemble-checkpoint checkpoints/ink_9um/hybrid_3d2d-seed43/step-075000.pth}
mkdir -p renders/$NAME pred/$NAME
for d in $(find -L "$SEGS" -name x.tif -exec dirname {} \; | sort); do
  n=$(basename $d)
  [ -f pred/$NAME/$n.tif ] && continue
  python render.py $d $VOL renders/$NAME/$n.zarr --flip "$@" 2>/dev/null || { echo "render fail $n"; continue; }
  python -m vesuvius.ink_detection.inference.infer renders/$NAME/$n.zarr $CK pred/$NAME/$n.tif \
    --overlap 0.5 --blend-mode hann --batch-size 16 --direction both --no-compile $ENS >/dev/null 2>&1 || echo "infer fail $n"
  python preview.py pred/$NAME/$n.tif pred/$NAME/${n}_reverse.tif renders/$NAME/$n.zarr_max.png
  rm -rf renders/$NAME/$n.zarr
  python prune_cache.py ${CHUNK_CACHE:-cache} ${CACHE_GB:-10} >/dev/null; echo "done $n"
done
