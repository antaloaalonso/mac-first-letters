#!/bin/bash
# usage: tta_run.sh <tag> <ckpt.pth> [extra infer flags...]
# TRANSPOSED=1 reads tta/<case>/<input>_T.zarr (tta_transpose.py make) and writes <input>_T_<tag>.tif.
# Example (the recommended screen): tta_run.sh ft42_mir ft9/ckpt/ft9_seed42.pth --direction forward --tta-mirror
# Runs one inference variant on the held-out RESULTS9 9.36 um inputs (mac9 = band-sweep render, off9 = official)
# and writes bench/tta/<case>/<input>_<tag>.tif (+ _reverse.tif with --direction both). Settings otherwise as
# run_infer9.sh / ft9/eval_ft.sh: --overlap 0.5 --blend-mode hann --batch-size 16 --no-compile.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); SP=$(cd "$HERE/.." && pwd)   # repo root
PY=$SP/.venv/bin/python
export PYTHONPATH=$SP/villa/vesuvius/src PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=4
TAG=$1; CK=$2; shift 2
for c in pherc0139-w016 pherc0814-46527; do
  mkdir -p $HERE/tta/$c
  for inp in mac9 off9; do
    if [ "${TRANSPOSED:-0}" = 1 ]; then src=$HERE/tta/$c/${inp}_T.zarr; out=$HERE/tta/$c/${inp}_T_$TAG.tif
    else src=$HERE/data/$c/$inp.zarr; out=$HERE/tta/$c/${inp}_$TAG.tif; fi
    [ -f $out ] && continue
    nice -n 15 $PY -m vesuvius.ink_detection.inference.infer $src $CK $out \
      --overlap 0.5 --blend-mode hann --batch-size 16 --no-compile "$@" > ${out%.tif}.log 2>&1
  done
done
