#!/bin/bash
# usage: run_infer9.sh <data_dir> <case>
# ink_9um full (seed42+43 ensemble) and screen50 (seed42 alone), both at 50% overlap, on off9.zarr and mac9.zarr.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); SP=$(cd "$HERE/.." && pwd)   # repo root
PY=$SP/.venv/bin/python; CK=$SP/checkpoints/ink_9um
export PYTHONPATH=$SP/villa/vesuvius/src PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=4
d=$1/$2; mkdir -p $d/pred9
for inp in ${INPUTS:-off9 mac9}; do
  for m in seed42 ens; do
    out=$d/pred9/${inp}_$m.tif; [ -f $out ] && continue
    if [ $m = ens ]; then
      args="$CK/hybrid_3d2d-seed42/step-075000.pth --ensemble-checkpoint $CK/hybrid_3d2d-seed43/step-075000.pth"
    else
      args="$CK/hybrid_3d2d-$m/step-075000.pth"
    fi
    nice -n 15 $PY -m vesuvius.ink_detection.inference.infer $d/$inp.zarr $args $out \
      --overlap 0.5 --blend-mode hann --batch-size 16 --direction forward --no-compile > $d/pred9/${inp}_$m.log 2>&1
    echo "$2 $inp $m done"
  done
done
