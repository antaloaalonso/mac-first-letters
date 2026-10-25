"""Stack winding ink predictions (one direction) into tall review sheets, outer windings first.
usage: montage.py <pred_dir> <forward|reverse> <out_prefix> [width] [per_sheet]"""
import cv2, glob, numpy as np, tifffile, sys, re
d, direction, out = sys.argv[1], sys.argv[2], sys.argv[3]
W = int(sys.argv[4]) if len(sys.argv) > 4 else 2000; per = int(sys.argv[5]) if len(sys.argv) > 5 else 6
fs = sorted(f for f in glob.glob(f'{d}/w*.tif') if f.endswith('_reverse.tif') == (direction == 'reverse'))
fs.sort(key=lambda f: -int(re.search(r'w(\d+)', f.split('/')[-1]).group(1)))
def norm(a):
    a = a.astype(np.float32); nz = a[a > 0]
    lo, hi = np.percentile(nz, 1), np.percentile(nz, 99.7)
    return (np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
import os
def void_mask(f, shape):
    # blank voids/cracks (dark in the surface composite) plus a margin: their rims light up the ink model
    name = os.path.basename(f).replace('_reverse.tif', '.tif').replace('.tif', '.zarr_max.png')
    p = os.path.join('renders', os.path.basename(d.rstrip('/')), name)
    if not os.path.exists(p): return np.ones(shape, bool)
    s = cv2.imread(p, 0)[:shape[0], :shape[1]].astype(np.float32)
    void = (s < 0.4 * np.median(s[s > 0])).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(void)
    keep = np.zeros(n, np.uint8); keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 400  # real voids, not fibre gaps
    void = cv2.dilate(keep[lab], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    m = np.ones(shape, bool); m[:s.shape[0], :s.shape[1]] = void == 0
    return m
for i in range(0, len(fs), per):
    rows = []
    for f in fs[i:i + per]:
        raw = tifffile.imread(f); a = norm(raw) * void_mask(f, raw.shape); s = W / a.shape[1]
        a = cv2.resize(a, (W, max(1, int(a.shape[0] * s))), interpolation=cv2.INTER_AREA)
        cv2.putText(a, re.search(r'w\d+', f.split('/')[-1]).group(0), (10, 40), 0, 1.3, 255, 3); rows.append(a)
    cv2.imwrite(f'{out}_{i // per}.jpg', np.vstack(rows)); print(f'{out}_{i // per}.jpg', [re.search(r"w\d+", f.split("/")[-1]).group(0) for f in fs[i:i + per]])
