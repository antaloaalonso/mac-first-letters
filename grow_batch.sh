#!/bin/bash
# usage: grow_batch.sh <surface.zarr> <out_dir> <params.json> <count>  — random non-overlapping GrowPatch seeds
cd "$(dirname "$0")"
T=tools/VC3D.app/Contents/MacOS
for i in $(seq 1 $4); do
  $T/vc_grow_seg_from_seed -v $1 -t $2 -p $3 >> logs_grow_batch.txt 2>&1
  echo "grown $i: $(ls $2 | wc -l) segments"
done
