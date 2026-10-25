#!/bin/bash
# usage: grow_seeds.sh <surface.zarr> <out_dir> <params.json> <seeds.txt> [n]  — GrowPatch from explicit seeds (x y z score), best first
cd "$(dirname "$0")"
T=tools/VC3D.app/Contents/MacOS
sort -k4 -nr "$4" | head -${5:-30} | while read x y z s; do
  $T/vc_grow_seg_from_seed -v $1 -t $2 -p $3 -s $x $y $z >> logs_grow_seeds.txt 2>&1
  echo "seed $x $y $z ($s): $(ls $2 | wc -l) segments"
done
