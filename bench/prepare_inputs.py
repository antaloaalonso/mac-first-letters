"""Build the two ink-model inputs for one validation case.

    python prepare_inputs.py <case> <data_dir> [--margin 256]

Both cover the case's validation box (from fetch_labels.py) plus a margin, on
the level-2 pixel grid the labels use:

  official.zarr  level 2 of the published 2.4 um surface volume, centred 84
                 planes mean-pooled by 4 to 21 (villa's prepare_9um_isotropic_input)
  mac.zarr       the published 2.399 um tifxyz, cropped, re-rendered by
                 ../render.py from the 2.4 um CT volume at
                 level 2 with 21 layers, streaming only the chunks it touches
"""
import argparse, io, json, math, os, subprocess, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, tifffile

S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
HERE = os.path.dirname(os.path.abspath(__file__))
RENDER = os.path.join(HERE, "..", "render.py")
CASES = {
    "pherc0139-w016": ("PHerc0139/segments/20250108000004-w029_2025010827", "20260102150214",
                       "PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr", "20250108000004"),
    "pherc0814-46527": ("PHerc0814/segments/20260226000000-46527_2um_try2", "20260309142202",
                        "PHerc0814/volumes/20260309142202-2.399um-0.2m-78keV-masked.zarr", "20260226000000"),
    "pherc1667-w029": ("PHerc1667/segments/20251212185248-w029_20251212185248662_flatboi", "20251217075048",
                       "PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr", "20251212185248"),
}


def write_zarr(path, vol):
    """Zarr v2 group with one uncompressed (Z, Y, X) uint8 array at '0'."""
    os.makedirs(os.path.join(path, "0"), exist_ok=True)
    json.dump({"zarr_format": 2}, open(os.path.join(path, ".zgroup"), "w"))
    z, h, w = vol.shape
    json.dump({"zarr_format": 2, "shape": [z, h, w], "chunks": [z, 128, 128], "dtype": "|u1",
               "compressor": None, "fill_value": 0, "order": "C", "filters": None,
               "dimension_separator": "/"}, open(os.path.join(path, "0", ".zarray"), "w"))
    for cy in range(-(-h // 128)):
        for cx in range(-(-w // 128)):
            b = np.zeros((z, 128, 128), np.uint8)
            s = vol[:, cy * 128:(cy + 1) * 128, cx * 128:(cx + 1) * 128]
            b[:, :s.shape[1], :s.shape[2]] = s
            d = os.path.join(path, "0", "0", str(cy)); os.makedirs(d, exist_ok=True)
            b.tofile(os.path.join(d, str(cx)))


def official(seg, vol_id, box, out):
    url = f"{S3}/{seg}/surface-volumes/2.399um-0.22m-78keV-volume-{vol_id}.zarr/2"
    meta = json.load(urllib.request.urlopen(url + "/.zarray"))
    Z, H, W = meta["shape"]; cz, cy, cx = meta["chunks"]
    assert meta["compressor"] is None and cz == Z
    y0, y1, x0, x1 = box
    crop = np.zeros((Z, y1 - y0, x1 - x0), np.uint8)

    def get(c):
        iy, ix = c
        try:
            raw = urllib.request.urlopen(f"{url}/0/{iy}/{ix}", timeout=120).read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return c, None
            raise
        return c, np.frombuffer(raw, np.uint8).reshape(cz, cy, cx)

    ids = [(iy, ix) for iy in range(max(0, y0) // cy, -(-min(H, y1) // cy))
           for ix in range(max(0, x0) // cx, -(-min(W, x1) // cx))]
    with ThreadPoolExecutor(16) as pool:
        for (iy, ix), a in pool.map(get, ids):
            if a is None:
                continue
            ys, xs = iy * cy, ix * cx
            sy0, sx0, sy1, sx1 = max(ys, y0), max(xs, x0), min(ys + cy, y1), min(xs + cx, x1)
            crop[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[:, sy0 - ys:sy1 - ys, sx0 - xs:sx1 - xs]
    # villa prepare_9um_isotropic_input: centred 84 planes, rounded mean of 4
    start = math.ceil((Z - 84) / 2)
    pooled = np.rint(crop[start:start + 84].astype(np.float32).reshape(21, 4, *crop.shape[1:]).mean(1))
    write_zarr(out, pooled.astype(np.uint8))


def mac(seg, vol_id, mesh_id, vol_path, box, out, work):
    mesh = f"{S3}/{seg}/mesh/{mesh_id}-on-{vol_id}-2.399um.tifxyz"
    meta = json.load(urllib.request.urlopen(mesh + "/meta.json"))
    per = round(1 / meta["scale"][0] / 4)  # level-2 pixels per mesh pixel
    y0, y1, x0, x1 = box
    my0, mx0 = max(0, y0 // per - 1), max(0, x0 // per - 1)
    my1, mx1 = -(-y1 // per) + 1, -(-x1 // per) + 1
    d = os.path.join(work, "mesh_crop"); os.makedirs(d, exist_ok=True)
    for a in "xyz":
        full = tifffile.imread(io.BytesIO(urllib.request.urlopen(f"{mesh}/{a}.tif").read()))
        tifffile.imwrite(os.path.join(d, f"{a}.tif"), full[my0:my1, mx0:mx1])
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
    raw = os.path.join(work, "mac_raw.zarr")
    subprocess.run([sys.executable, RENDER, d, f"{S3}/{vol_path}", raw, "--flip", "--level", "2",
                    "--frame-level", "0", "--layers", "21"], check=True)
    import zarr
    full = np.asarray(zarr.open(raw + "/0", mode="r"))
    oy, ox = y0 - my0 * per, x0 - mx0 * per
    crop = np.zeros((21, y1 - y0, x1 - x0), np.uint8)
    part = full[:, max(0, oy):oy + (y1 - y0), max(0, ox):ox + (x1 - x0)]
    crop[:, max(0, -oy):max(0, -oy) + part.shape[1], max(0, -ox):max(0, -ox) + part.shape[2]] = part
    write_zarr(out, crop)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case"); ap.add_argument("data_dir"); ap.add_argument("--margin", type=int, default=256)
    ap.add_argument("--only", choices=("official", "mac"))
    a = ap.parse_args()
    seg, vol_id, vol_path, mesh_id = CASES[a.case]
    cdir = os.path.join(a.data_dir, a.case)
    lab = np.load(os.path.join(cdir, "labels.npz"))
    H, W = [int(v) for v in lab["shape"]]
    b = [int(v) for v in lab["box"]]
    box = (max(0, b[0] - a.margin), min(H, b[1] + a.margin), max(0, b[2] - a.margin), min(W, b[3] + a.margin))
    json.dump({"input_box": box, "label_box": b}, open(os.path.join(cdir, "boxes.json"), "w"))
    if a.only != "mac":
        official(seg, vol_id, box, os.path.join(cdir, "official.zarr")); print(a.case, "official done", box)
    if a.only != "official":
        mac(seg, vol_id, mesh_id, vol_path, box, os.path.join(cdir, "mac.zarr"), cdir); print(a.case, "mac done", box)
