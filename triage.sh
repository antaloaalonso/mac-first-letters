#!/bin/bash
# usage: triage.sh <segments_dir> <volume_url> <frame_level>  — cheap level-3 previews + contact sheet (triage/<dir>.jpg)
cd "$(dirname "$0")"; source .venv/bin/activate
name=$(basename $1); mkdir -p triage/$name
for d in $(find -L $1 -name x.tif -exec dirname {} \; | sort); do
  n=$(basename $d)
  [ -f triage/$name/$n.zarr_max.png ] || python render.py $d $2 triage/$name/$n.zarr --flip --level 3 --frame-level $3 --layers 13 2>/dev/null
  rm -rf triage/$name/$n.zarr
done
python - $name <<'PY'
import cv2, glob, numpy as np, sys
name = sys.argv[1]; ims = []
for f in sorted(glob.glob(f'triage/{name}/*_max.png')):
    a = cv2.resize(cv2.imread(f, 0), (400, 400)); n = f.split('/')[-1].split('.')[0]
    cv2.putText(a, n[-9:], (5, 25), 0, 0.8, 255, 2); ims.append(a)
while len(ims) % 5: ims.append(np.zeros((400, 400), np.uint8))
cv2.imwrite(f'triage/{name}.jpg', np.vstack([np.hstack(ims[i:i + 5]) for i in range(0, len(ims), 5)]))
PY
