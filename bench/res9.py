"""Does ink_9um still find held-out ink on the 9.36 um scans, as the band sweep uses them?

    python res9.py prepare <data_dir> <case>    # 9.36 um inputs + registration onto the label grid
    python res9.py score   <data_dir> <case>    # warp 9.36 um predictions onto the labels and score

For a validation case that also has a 9.362 um scan, `prepare` writes, over the label box:
  off9.zarr    the published 9.362 um surface volume (level 0, 28 planes)
  mac9.zarr    the published 9.362 um tifxyz re-rendered by render.py at level 0 with 31
               layers, the band sweep's input
and fits an affine map from the 9.36 um surface grid to the label grid (2.4 um level 2) by ECC
on the centre layers of off9 and official.zarr. `score` warps each pred9/<input>_<model>.tif
into the label grid and scores it inside the validation mask like score.py.
"""
import io, json, os, subprocess, sys, urllib.request
import numpy as np, tifffile, cv2, zarr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prepare_inputs import S3, RENDER, write_zarr
from score import roc_pr

CASES = {
    "pherc0139-w016": ("PHerc0139/segments/20250108000004-w029_2025010827", "20250728140407",
                       "PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr", "20250108000004"),
    "pherc0814-46527": ("PHerc0814/segments/20260226000000-46527_2um_try2", "20250804134230",
                        "PHerc0814/volumes/20250804134230-9.362um-1.2m-113keV-masked.zarr", "20260226000000"),
}
MARGIN = 400


def read_crop(url, box):
    meta = json.load(urllib.request.urlopen(url + "/.zarray"))
    Z, H, W = meta["shape"]; cz, cy, cx = meta["chunks"]
    y0, y1, x0, x1 = max(0, box[0]), min(H, box[1]), max(0, box[2]), min(W, box[3])
    out = np.zeros((Z, y1 - y0, x1 - x0), np.uint8)
    for iy in range(y0 // cy, -(-y1 // cy)):
        for ix in range(x0 // cx, -(-x1 // cx)):
            try:
                a = np.frombuffer(urllib.request.urlopen(f"{url}/0/{iy}/{ix}", timeout=120).read(), np.uint8).reshape(cz, cy, cx)
            except urllib.error.HTTPError:
                continue
            ys, xs = iy * cy, ix * cx
            sy0, sx0, sy1, sx1 = max(ys, y0), max(xs, x0), min(ys + cy, y1), min(xs + cx, x1)
            out[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[:, sy0 - ys:sy1 - ys, sx0 - xs:sx1 - xs]
    return out, (y0, y1, x0, x1)


def prepare(data_dir, case):
    seg, vol9, vol_path, mesh_id = CASES[case]
    cdir = os.path.join(data_dir, case)
    boxes = json.load(open(os.path.join(cdir, "boxes.json")))
    iy0, iy1, ix0, ix1 = boxes["input_box"]
    # 9.362 um pixel = 9.362 / 9.596 label pixels; take the matching box plus a margin
    s = 9.596 / 9.362
    box9 = (int(iy0 * s) - MARGIN, int(iy1 * s) + MARGIN, int(ix0 * s) - MARGIN, int(ix1 * s) + MARGIN)
    off9, box9 = read_crop(f"{S3}/{seg}/surface-volumes/9.362um-1.2m-113keV-volume-{vol9}.zarr/0", box9)
    write_zarr(os.path.join(cdir, "off9.zarr"), off9)

    # the sweep's input: the 9.362 um mesh, cropped, rendered at level 0, 31 layers
    mesh = f"{S3}/{seg}/mesh/{mesh_id}-on-{vol9}-9.362um.tifxyz"
    meta = json.load(urllib.request.urlopen(mesh + "/meta.json"))
    per = round(1 / meta["scale"][0])
    my0, my1, mx0, mx1 = box9[0] // per, -(-box9[1] // per) + 1, box9[2] // per, -(-box9[3] // per) + 1
    d = os.path.join(cdir, "mesh9_crop"); os.makedirs(d, exist_ok=True)
    for a in "xyz":
        full = tifffile.imread(io.BytesIO(urllib.request.urlopen(f"{mesh}/{a}.tif").read()))
        tifffile.imwrite(os.path.join(d, f"{a}.tif"), full[my0:my1, mx0:mx1])
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
    raw = os.path.join(cdir, "mac9_raw.zarr")
    subprocess.run([sys.executable, RENDER, d, f"{S3}/{vol_path}", raw, "--flip", "--level", "0",
                    "--frame-level", "0", "--layers", "31"], check=True)
    full = np.asarray(zarr.open(raw + "/0", mode="r"))
    oy, ox = box9[0] - my0 * per, box9[2] - mx0 * per
    mac9 = np.zeros((31,) + off9.shape[1:], np.uint8)
    part = full[:, oy:oy + off9.shape[1], ox:ox + off9.shape[2]]
    mac9[:, :part.shape[1], :part.shape[2]] = part
    write_zarr(os.path.join(cdir, "mac9.zarr"), mac9)

    # register: label-grid coords = A @ (9 um coords) + t, fitted on the centre layers
    ref = np.asarray(zarr.open(os.path.join(cdir, "official.zarr", "0"), mode="r"))[10].astype(np.float32)
    mov = off9[off9.shape[0] // 2].astype(np.float32)
    # initial guess: scale 9.362/9.596, and the crop offsets
    sc = 1 / s
    warp = np.array([[sc, 0, box9[2] * sc - ix0], [0, sc, box9[0] * sc - iy0]], np.float32)
    def prep(a):
        a = cv2.GaussianBlur(a, (0, 0), 2); m = a > 0
        return np.where(m, (a - a[m].mean()) / (a[m].std() + 1e-6), 0).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-7)
    # ECC warps the template (ref) into the input (mov) frame; fit mov -> ref by inverting roles
    inv = cv2.invertAffineTransform(warp)
    cc, inv = cv2.findTransformECC(prep(ref), prep(mov), inv.astype(np.float32), cv2.MOTION_AFFINE, crit, None, 5)
    warp = cv2.invertAffineTransform(inv)
    reg = cv2.warpAffine(mov, warp, ref.shape[::-1], flags=cv2.INTER_LINEAR)
    m = (reg > 0) & (ref > 0)
    ncc = float(np.corrcoef(reg[m], ref[m])[0, 1])
    mac_c = mac9[15].astype(np.float32); mm = (mac_c > 0) & (mov > 0)
    out = {"box9": box9, "warp_9_to_label_input": warp.tolist(), "ecc": float(cc), "ncc_off9_vs_off24": ncc,
           "ncc_mac9_vs_off9": float(np.corrcoef(mac_c[mm], mov[mm])[0, 1])}
    json.dump(out, open(os.path.join(cdir, "res9.json"), "w"), indent=1)
    print(case, json.dumps(out))


def score(data_dir, case):
    cdir = os.path.join(data_dir, case)
    reg = json.load(open(os.path.join(cdir, "res9.json")))
    warp = np.array(reg["warp_9_to_label_input"], np.float32)
    boxes = json.load(open(os.path.join(cdir, "boxes.json")))
    iy0, iy1, ix0, ix1 = boxes["input_box"]; ly0, ly1, lx0, lx1 = boxes["label_box"]
    lab = np.load(os.path.join(cdir, "labels.npz"))
    valid, ink = lab["validation_mask"] > 0, lab["inklabels"] > 0
    rows = []
    for f in sorted(os.listdir(os.path.join(cdir, "pred9"))):
        if not f.endswith(".tif") or f.endswith("_reverse.tif"):
            continue
        p = tifffile.imread(os.path.join(cdir, "pred9", f)).astype(np.float32)
        w = cv2.warpAffine(p, warp, (ix1 - ix0, iy1 - iy0), flags=cv2.INTER_LINEAR, borderValue=np.nan)
        crop = w[ly0 - iy0:ly1 - iy0, lx0 - ix0:lx1 - ix0]
        m = valid & np.isfinite(crop)
        r = {"case": case, "pred": f, "pixels": int(m.sum()), "of_valid": int(valid.sum()), **roc_pr(crop[m], ink[m])}
        rows.append(r); print(json.dumps(r))
    json.dump(rows, open(os.path.join(cdir, "res9_scores.json"), "w"), indent=1)


def control(data_dir, case):
    """Resampling control: push the 2.4 um predictions through the same warp to the 9.36 um grid
    and back, then score. The drop it causes is what registration and interpolation alone cost."""
    cdir = os.path.join(data_dir, case)
    reg = json.load(open(os.path.join(cdir, "res9.json")))
    warp = np.array(reg["warp_9_to_label_input"], np.float32)
    b9 = reg["box9"]; shape9 = (b9[3] - b9[2], b9[1] - b9[0])
    boxes = json.load(open(os.path.join(cdir, "boxes.json")))
    iy0, iy1, ix0, ix1 = boxes["input_box"]; ly0, ly1, lx0, lx1 = boxes["label_box"]
    lab = np.load(os.path.join(cdir, "labels.npz"))
    valid, ink = lab["validation_mask"] > 0, lab["inklabels"] > 0
    rows = []
    for f in ("official_ens.tif", "official_seed42.tif", "mac_ens.tif", "mac_seed42.tif"):
        p = tifffile.imread(os.path.join(cdir, "pred", f)).astype(np.float32)
        to9 = cv2.warpAffine(p, cv2.invertAffineTransform(warp), shape9, flags=cv2.INTER_LINEAR, borderValue=np.nan)
        back = cv2.warpAffine(to9, warp, (ix1 - ix0, iy1 - iy0), flags=cv2.INTER_LINEAR, borderValue=np.nan)
        crop = back[ly0 - iy0:ly1 - iy0, lx0 - ix0:lx1 - ix0]
        m = valid & np.isfinite(crop)
        r = {"case": case, "pred": "roundtrip_" + f, "pixels": int(m.sum()), "of_valid": int(valid.sum()),
             **roc_pr(crop[m], ink[m])}
        rows.append(r); print(json.dumps(r))
    json.dump(rows, open(os.path.join(cdir, "res9_control.json"), "w"), indent=1)


if __name__ == "__main__":
    {"prepare": prepare, "score": score, "control": control}[sys.argv[1]](sys.argv[2], sys.argv[3])
