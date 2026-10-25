"""Contact images for ink predictions: forward | reverse | surface, downscaled. usage: view.py <pred_dir> [width]"""
import cv2, glob, numpy as np, tifffile, sys, os
d = sys.argv[1]; W = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
def norm(a):
    a = a.astype(np.float32); nz = a[a > 0]
    if not nz.size: return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(nz, 1), np.percentile(nz, 99.7)
    return (np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
for f in sorted(glob.glob(f'{d}/*.tif')):
    if f.endswith('_reverse.tif'): continue
    out = f.replace('.tif', '_view.jpg')
    if os.path.exists(out): continue
    ims = [norm(tifffile.imread(f)), norm(tifffile.imread(f.replace('.tif', '_reverse.tif')))]
    s = (W / 3) / max(ims[0].shape)
    ims = [cv2.resize(i, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for i in ims]
    cv2.imwrite(out, np.hstack(ims), [cv2.IMWRITE_JPEG_QUALITY, 90]); print(out)
