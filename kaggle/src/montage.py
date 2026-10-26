"""Stack winding ink predictions (one direction) into tall review sheets, outer windings first.
Port of ../montage.py with explicit paths: void/crack rims light up the ink model, so voids
(dark in the surface max-composite) are blanked with a 25 px margin before review."""
import glob, os, re
import cv2, numpy as np, tifffile


def norm(a):
    a = a.astype(np.float32); nz = a[a > 0]
    if not nz.size:
        return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(nz, 1), np.percentile(nz, 99.7)
    return (np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


def void_mask(max_png, shape):
    if not os.path.exists(max_png):
        return np.ones(shape, bool)
    s = cv2.imread(max_png, 0)[:shape[0], :shape[1]].astype(np.float32)
    if not (s > 0).any():
        return np.ones(shape, bool)
    void = (s < 0.4 * np.median(s[s > 0])).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(void)
    keep = np.zeros(n, np.uint8); keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 400  # real voids, not fibre gaps
    void = cv2.dilate(keep[lab], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    m = np.ones(shape, bool); m[:s.shape[0], :s.shape[1]] = void == 0
    return m


def winding_no(f):
    return int(re.search(r"w(\d+)", os.path.basename(f)).group(1))


def montage(pred_dir, render_dir, direction, out_prefix, width=3000, per=6, suffix=".tif"):
    fs = sorted(f for f in glob.glob(f"{pred_dir}/w*{suffix}")
                if f.endswith(f"_reverse{suffix}") == (direction == "reverse"))
    fs.sort(key=lambda f: -winding_no(f))
    outs = []
    for i in range(0, len(fs), per):
        rows = []
        for f in fs[i:i + per]:
            raw = tifffile.imread(f)
            name = re.search(r"w\d+", os.path.basename(f)).group(0)
            a = norm(raw) * void_mask(os.path.join(render_dir, f"{name}_max.png"), raw.shape)
            s = width / a.shape[1]
            a = cv2.resize(a, (width, max(1, int(a.shape[0] * s))), interpolation=cv2.INTER_AREA)
            cv2.putText(a, name, (10, 40), 0, 1.3, 255, 3); rows.append(a)
        p = f"{out_prefix}_{i // per}.jpg"
        cv2.imwrite(p, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
        outs.append(p)
    return outs


def masked_fullres(pred_tif, max_png, out_jpg):
    """Full-resolution masked JPG of one prediction, for reading letters at 1:1."""
    raw = tifffile.imread(pred_tif)
    cv2.imwrite(out_jpg, norm(raw) * void_mask(max_png, raw.shape), [cv2.IMWRITE_JPEG_QUALITY, 88])
