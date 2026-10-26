#!/bin/bash
# usage: eval_ft.sh <tag> <ckpt.pth> [ensemble_ckpt.pth]
# Runs a (fine-tuned) ink_9um checkpoint at 50% overlap on the held-out RESULTS9 inputs:
# off9.zarr / mac9.zarr (9.362 um) and official.zarr (2.4 um L2), writing pred9/<input>_<tag>.tif
# and pred/official_<tag>.tif, then scores them (res9.py score, score.py).
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd); SP=$(cd "$HERE/.." && pwd)   # repo root
PY=$SP/.venv/bin/python
export PYTHONPATH=$SP/villa/vesuvius/src PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=4
TAG=$1; CK=$2; ENS=${3:+--ensemble-checkpoint $3}
for c in pherc0139-w016 pherc0814-46527; do
  d=$HERE/data/$c
  for inp in off9 mac9 official; do
    if [ $inp = official ]; then out=$d/pred/official_$TAG.tif; else out=$d/pred9/${inp}_$TAG.tif; fi
    [ -f $out ] && continue
    nice -n 15 $PY -m vesuvius.ink_detection.inference.infer $d/$inp.zarr $CK $ENS $out \
      --overlap 0.5 --blend-mode hann --batch-size 16 --direction forward --no-compile > ${out%.tif}.log 2>&1
  done
  $PY $HERE/res9.py score $HERE/data $c | grep "_$TAG.tif"
  $PY $HERE/score.py $HERE/data $c $d/pred/official_$TAG.tif
done
