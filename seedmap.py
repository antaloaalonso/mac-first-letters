"""Score surface-prediction slices for clean, thin, parallel sheets and emit GrowPatch seeds.

    python seedmap.py <surface.zarr> <out_seeds.txt> [--z 900,1400,...] [--per-z 12]
Score per 128-px block = orientation coherence of the sheet mask x thinness
(skeleton pixels / mask pixels), restricted to blocks with moderate sheet density.
"""
import argparse, numpy as np, zarr, cv2
from skimage.morphology import skeletonize

ap = argparse.ArgumentParser()
ap.add_argument("zarr"); ap.add_argument("out")
ap.add_argument("--z", default="700,1100,1500,1900,2300,2700,3100")
ap.add_argument("--per-z", type=int, default=12)
ap.add_argument("--block", type=int, default=128)
a = ap.parse_args()
arr = zarr.open(a.zarr + "/0", mode="r")
B = a.block
out, vis = [], []
for z in map(int, a.z.split(",")):
    m = (np.asarray(arr[z]) > 0).astype(np.uint8)
    sk = skeletonize(m > 0).astype(np.float32)
    g = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 3)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    jxx, jyy, jxy = [cv2.boxFilter(v, -1, (B, B)) for v in (gx * gx, gy * gy, gx * gy)]
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    H, W = m.shape
    scores = []
    for y in range(B, H - B, B):
        for x in range(B, W - B, B):
            mb, sb = m[y:y + B, x:x + B], sk[y:y + B, x:x + B]
            dens = sb.sum() / B  # sheet crossings per block row
            if not 3 <= dens <= 25:
                continue
            thin = sb.sum() / max(mb.sum(), 1)
            s = coh[y + B // 2, x + B // 2] * min(1, thin * 3)
            scores.append((s, x, y))
    scores.sort(reverse=True)
    picked = []
    for s, x, y in scores:
        if len(picked) >= a.per_z: break
        if any(abs(x - px) < 4 * B and abs(y - py) < 4 * B for _, px, py in picked):
            continue  # spread seeds out
        ys, xs = np.nonzero(sk[y:y + B, x:x + B])
        i = np.argmin((ys - B / 2) ** 2 + (xs - B / 2) ** 2)
        picked.append((s, x, y)); out.append((x + xs[i], y + ys[i], z, s))
    v = cv2.cvtColor(cv2.resize(m * 255, (W // 4, H // 4)), cv2.COLOR_GRAY2BGR)
    for s, x, y in picked:
        cv2.circle(v, ((x + B // 2) // 4, (y + B // 2) // 4), 12, (0, 0, 255), 3)
    cv2.putText(v, f"z={z}", (20, 60), 0, 2, (0, 255, 0), 4); vis.append(v)
with open(a.out, "w") as f:
    for x, y, z, s in out: f.write(f"{x} {y} {z} {s:.3f}\n")
cv2.imwrite(a.out.replace(".txt", ".jpg"), np.vstack([np.hstack(vis[i:i + 4]) if len(vis[i:i + 4]) == 4 else np.hstack(vis[i:i + 4] + [np.zeros_like(vis[0])] * (4 - len(vis[i:i + 4]))) for i in range(0, len(vis), 4)]))
print(len(out), "seeds")
