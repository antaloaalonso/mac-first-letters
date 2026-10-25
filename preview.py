"""Side-by-side JPG: surface max-composite | ink forward | ink reverse (percentile-normalised)."""
import sys, numpy as np, tifffile, cv2
ims = []
for f in sys.argv[1:]:
    a = (cv2.imread(f, cv2.IMREAD_GRAYSCALE) if f.endswith('.png') else tifffile.imread(f)).astype(np.float32)
    nz = a[a > 0]
    if nz.size:
        lo, hi = np.percentile(nz, 1), np.percentile(nz, 99.7)
        a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * (a > 0)
    ims.append((a * 255).astype(np.uint8))
h = min(i.shape[0] for i in ims); w = min(i.shape[1] for i in ims)
out = np.hstack([i[:h, :w] for i in ims])
cv2.imwrite(sys.argv[1].replace('.tif', '_cmp.jpg'), out, [cv2.IMWRITE_JPEG_QUALITY, 88])
