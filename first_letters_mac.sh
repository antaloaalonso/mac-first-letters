#!/bin/bash
# One command from a scroll's published representations to ink review sheets, on a Mac.
#
#   ./first_letters_mac.sh <scroll> <volume_id> <z0> <z1>
#   e.g. ./first_letters_mac.sh PHerc0846A 20260319102732 1600 1800
#
# <volume_id> names a 2.4 um scan with published surface-m7 level-2 predictions and
# normal grids; z0..z1 is the band in that level-2 frame (9.6 um voxels).
# Stages, each skipped when its output exists (so a run resumes):
#   fetch      surface prediction + normal grids          -> data/<scroll>_<volume_id>/
#   umbilicus  vc_gen_umbilicus on the normal grids, median-smoothed
#   patches    seedmap.py seeds in the band + GrowPatch    (the fit's verified patches)
#   tracks     extract_surface_tracks.py --backend cpu over the band
#   fit        fit_spiral.py on $DEVICE (default mps)      -> one tifxyz per winding
#   ink        run_segments.sh: stream-render each winding at level 2, ink_9um
#              seed42+43 ensemble, both directions
#   montage    void-masked review sheets per direction     -> work/<run>/montage_*.jpg
# Knobs (environment): STEPS=3000 SEEDS_PER_Z=4 OUTWARD=CW NUM_WINDINGS=90
#   DEVICE=mps MAX_WINDINGS=all DATA_DIR=... WORK=work CACHE_GB=10
set -euo pipefail
[ $# -eq 4 ] || { sed -n 2,20p "$0"; exit 2; }
SCROLL=$1; VOL_ID=$2; Z0=$3; Z1=$4
ROOT=$(cd "$(dirname "$0")" && pwd); cd "$ROOT"
S3=https://vesuvius-challenge-open-data.s3.amazonaws.com
PY=$ROOT/.venv/bin/python; SPY=$ROOT/villa/spiral-fitting/.venv/bin/python
VC=$ROOT/tools/VC3D.app/Contents/MacOS
STEPS=${STEPS:-3000}; DEVICE=${DEVICE:-mps}; SEEDS_PER_Z=${SEEDS_PER_Z:-4}
DATA=${DATA_DIR:-$ROOT/data/${SCROLL}_$VOL_ID}
RUN=${SCROLL}_${Z0}_${Z1}; W=$ROOT/${WORK:-work}/$RUN; DS=$W/dataset
mkdir -p "$W" "$DS/tracks" "$DS/verified_patches"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} CHUNK_CACHE=${CHUNK_CACHE:-$ROOT/cache}
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$W/pipeline.log"; }
run() { nice -n 15 "$@"; }

VOL_URL=$($PY s3ls.py "$SCROLL/volumes/" | awk -v id="$VOL_ID" '$2 ~ id {print $2}' | head -1)
[ -n "$VOL_URL" ] || { echo "no volume $VOL_ID under $SCROLL/volumes/"; exit 1; }
VOL_URL=$S3/${VOL_URL%/}

# --- fetch -------------------------------------------------------------------
if [ ! -d "$DATA/surface.zarr/0" ] || [ ! -d "$DATA/normal-grids/xy" ]; then
  log "fetch: surface prediction + normal grids for $VOL_ID"
  base=$($PY s3ls.py "$SCROLL/representations/predictions/surfaces/" | awk -v id="$VOL_ID" '$2 ~ id && $2 ~ /-L2-/ {print $2}')
  pred=$(echo "$base" | grep '\.zarr/$'); grids=$(echo "$base" | grep 'normal-grids/$')
  run $PY fetch_prefix.py "$pred" "$DATA/surface.zarr/"
  run $PY fetch_prefix.py "$grids" "$DATA/normal-grids/"
  # vc_grow_seg_from_seed rejects the published metadata.json (no "scan" key)
  [ -f "$DATA/surface.zarr/metadata.json" ] && mv "$DATA/surface.zarr/metadata.json" "$DATA/surface.zarr/metadata.published.json"
fi
read -r NZ NY NX < <($PY -c "import json;print(*json.load(open('$DATA/surface.zarr/0/.zarray'))['shape'])")
VOXEL=$($PY -c "import json;print(round(4*float(json.load(open('$DATA/surface.zarr/.zattrs')).get('voxel_size_um',2.403)),3))" 2>/dev/null || echo 9.612)

# --- umbilicus ---------------------------------------------------------------
if [ ! -f "$DS/umbilicus.json" ]; then
  log "umbilicus: vc_gen_umbilicus over $NZ slices"
  run $VC/vc_gen_umbilicus -i "$DATA/normal-grids" -o "$DS/umbilicus.json" --slices 100:$((NZ - 50)):100 \
    --seed 1 --repeats 3 --threads 8 --voxelsize-um "$VOXEL" --volume-width "$NX" --volume-height "$NY" --volume-slices "$NZ"
  $PY smooth_umbilicus.py "$DS/umbilicus.json" 5
fi

# --- patches -----------------------------------------------------------------
if [ -z "$(ls -A "$DS/verified_patches")" ]; then
  log "patches: seeds in [$Z0,$Z1) + GrowPatch"
  zs=$(( Z0 + (Z1 - Z0) / 4 )),$(( (Z0 + Z1) / 2 )),$(( Z1 - (Z1 - Z0) / 4 ))
  run $PY seedmap.py "$DATA/surface.zarr" "$W/seeds.txt" --z "$zs" --per-z "$SEEDS_PER_Z"
  $PY - "$W/grow_params.json" "$DATA/normal-grids" "$W/grow_cache" "$VOXEL" <<'PY'
import json, sys
out, grids, cache, vox = sys.argv[1:]
json.dump({"mode": "explicit_seed", "generations": 50, "min_area_cm": 1.0, "thread_limit": 4,
           "use_cuda": False, "voxelsize": float(vox), "normal_grid_path": grids, "cache_root": cache},
          open(out, "w"), indent=1)
PY
  run ./grow_seeds.sh "$DATA/surface.zarr" "$W/patches" "$W/grow_params.json" "$W/seeds.txt" 99
  for p in "$W"/patches/*/; do ln -sfn "${p%/}" "$DS/verified_patches/$(basename "$p")"; done
fi

# --- tracks ------------------------------------------------------------------
if [ ! -f "$DS/tracks/surf.dbm.vctracks" ]; then
  log "tracks: extract_surface_tracks.py --backend cpu over [$Z0,$Z1)"
  (cd villa/spiral-fitting && run $SPY extract_surface_tracks.py --predictions "$DATA/surface.zarr/0" \
     --out "$DS/tracks/surf.dbm" --z-min "$Z0" --z-max "$Z1" --backend cpu) > "$W/tracks.log" 2>&1
fi
cat > "$DS/spiral-scroll.json" <<EOF
{"schema_version": 1, "name": "$SCROLL", "voxel_size_um": $VOXEL,
 "spiral_outward_sense": "${OUTWARD:-CW}", "num_windings": ${NUM_WINDINGS:-90},
 "paths": {"tracks_dbm": "tracks/surf.dbm"}}
EOF

# --- fit ---------------------------------------------------------------------
FIT=$W/fit
if [ ! -d "$FIT/meshes/fitted" ]; then
  log "fit: $STEPS steps on $DEVICE"
  source spiral/fit_env.sh
  BASE=$(echo "$BASE" | sed "s/\"z_begin\":[0-9]*/\"z_begin\":$Z0/; s/\"z_end\":[0-9]*/\"z_end\":$Z1/")
  export FIT_SPIRAL_DEVICE=$DEVICE FIT_SPIRAL_RUN_DIR=$FIT FIT_SPIRAL_CACHE_DIR=$W/fit_cache
  export FIT_SPIRAL_CONFIG_OVERRIDES="{$BASE,\"optimizer_num_training_steps\":$STEPS}"
  [ -f "$FIT/checkpoint_fitted.ckpt" ] && export FIT_SPIRAL_RESUME_PATH=$FIT/checkpoint_fitted.ckpt
  (cd villa/spiral-fitting && run $SPY fit_spiral.py --dataset "$DS") > "$W/fit.log" 2>&1
fi

# --- ink ---------------------------------------------------------------------
mkdir -p "$W/windings"
ls -d "$FIT"/meshes/fitted/w[0-9][0-9][0-9] | sort -r | head -n "${MAX_WINDINGS:-100000}" |
  while read -r d; do ln -sfn "$d" "$W/windings/$(basename "$d")"; done
log "ink: $(ls "$W/windings" | wc -l | tr -d ' ') windings from $VOL_URL"
CACHE_GB=${CACHE_GB:-10} run ./run_segments.sh "$W/windings" "$VOL_URL" "$RUN" --level 2 --frame-level 2 | tee -a "$W/pipeline.log"

# --- montage -----------------------------------------------------------------
for dir in forward reverse; do run $PY montage.py "pred/$RUN" "$dir" "$W/montage_$dir" 2000 6 | tee -a "$W/pipeline.log"; done
log "done: review $W/montage_forward_*.jpg and $W/montage_reverse_*.jpg"
