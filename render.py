"""Render a tifxyz surface into a flat surface-volume zarr, streaming the scroll
volume straight from the open-data S3 bucket (uncompressed 128^3 zarr v2 chunks).

    python render.py <tifxyz_dir> <volume_url> <out.zarr> [--layers 31] [--level 0]

Coordinates in x/y/z.tif are level-0 voxels; output matches the layout of the
published segment surface volumes ([layers, H, W] uint8, chunks [layers,128,128]).
"""
import argparse, json, os, sys, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, tifffile, cv2
from scipy.ndimage import map_coordinates

CACHE = os.environ.get("CHUNK_CACHE", os.path.join(os.path.dirname(__file__), "cache"))


class RemoteVolume:
    def __init__(self, url, level=0):
        self.url = f"{url.rstrip('/')}/{level}"
        meta = json.loads(urllib.request.urlopen(self.url + "/.zarray").read())
        assert meta["compressor"] is None and meta["dtype"] == "|u1"
        self.shape, self.cs = meta["shape"], meta["chunks"][0]
        key = url.rstrip("/").split("/")
        self.cache = os.path.join(CACHE, key[-3] if len(key) > 2 else "", key[-1], str(level))
        self.lock = threading.Lock()

    def chunk(self, cz, cy, cx):
        p = os.path.join(self.cache, f"{cz}/{cy}/{cx}")
        if os.path.exists(p):
            raw = open(p, "rb").read()
        else:
            for attempt in range(6):
                try:
                    raw = urllib.request.urlopen(f"{self.url}/{cz}/{cy}/{cx}", timeout=60).read()
                    break
                except urllib.error.HTTPError as e:
                    if e.code in (403, 404):
                        raw = b""  # masked / absent chunk = fill 0
                        break
                except Exception:
                    if attempt == 5:
                        raise
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p + ".part", "wb").write(raw)
            os.replace(p + ".part", p)
        c = self.cs
        if not raw:
            return np.zeros((c, c, c), np.uint8)
        return np.frombuffer(raw, np.uint8).reshape(c, c, c)

    def read(self, lo, hi, pool):
        """Dense crop [lo, hi) in (z, y, x) voxels, zeros outside."""
        lo = [max(0, int(v)) for v in lo]
        hi = [min(s, int(v)) for v, s in zip(hi, self.shape)]
        out = np.zeros([max(0, h - l) for l, h in zip(lo, hi)], np.uint8)
        if out.size == 0:
            return out, lo
        c = self.cs
        idx = [(z, y, x) for z in range(lo[0] // c, (hi[0] - 1) // c + 1)
               for y in range(lo[1] // c, (hi[1] - 1) // c + 1)
               for x in range(lo[2] // c, (hi[2] - 1) // c + 1)]
        for (z, y, x), ch in zip(idx, pool.map(lambda t: self.chunk(*t), idx)):
            sl_o, sl_c = [], []
            for a, b in zip((z, y, x), range(3)):
                s0, s1 = max(lo[b], a * c), min(hi[b], (a + 1) * c)
                sl_o.append(slice(s0 - lo[b], s1 - lo[b]))
                sl_c.append(slice(s0 - a * c, s1 - a * c))
            out[tuple(sl_o)] = ch[tuple(sl_c)]
        return out, lo


def load_tifxyz(d, step=None):
    xyz = np.stack([tifffile.imread(os.path.join(d, f"{a}.tif")).astype(np.float32) for a in "xyz"], -1)
    meta = json.load(open(os.path.join(d, "meta.json")))
    valid = (xyz[..., 2] > 0) & np.isfinite(xyz).all(-1)
    up = step or round(1 / meta["scale"][0])
    H, W = xyz.shape[:2]
    # upsample to ~1 px per voxel, output pixel j <-> mesh coordinate j / up
    # (the published surface volumes' grid; cv2.resize's pixel-centre
    # convention would shift everything by (up - 1) / 2 px); invalid
    # neighbours are masked out afterwards
    gy, gx = np.meshgrid(np.arange(H * up, dtype=np.float32) / up,
                         np.arange(W * up, dtype=np.float32) / up, indexing="ij")
    big = cv2.remap(np.where(valid[..., None], xyz, 0), gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    vm = cv2.remap(valid.astype(np.float32), gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) > 0.999
    return big, vm


def normals(p):
    du = np.gradient(p, axis=1)
    dv = np.gradient(p, axis=0)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    return n


def render(tifxyz, vol_url, out, layers=31, level=0, tile=256, flip=False, frame_level=0, pool_k=1):
    """pool_k>1: sample k-times finer (layers*k planes) and mean-pool k x k x k, mimicking
    the 2.4 um level2/zmean4 recipe the ink models were trained on."""
    sc = 2 ** (level - frame_level)  # mesh coords are in the frame_level grid
    meta = json.load(open(os.path.join(tifxyz, "meta.json")))
    # one output pixel (and one layer step) = one voxel of the sampled level
    pts, vm = load_tifxyz(tifxyz, step=max(1, round(1 / meta["scale"][0] / sc)))
    n = normals(pts) * (-1 if flip else 1)
    vol = RemoteVolume(vol_url, level)
    k = pool_k
    Hf, Wf = (vm.shape[0] // k) * k, (vm.shape[1] // k) * k
    pts, vm = pts[:Hf, :Wf], vm[:Hf, :Wf]
    n = n[:Hf, :Wf]
    H, W = Hf // k, Wf // k
    half = layers // 2
    os.makedirs(os.path.join(out, "0"), exist_ok=True)
    json.dump({"zarr_format": 2}, open(os.path.join(out, ".zgroup"), "w"))
    json.dump({"multiscales": [{"version": "0.4", "axes": [{"name": a, "type": "space"} for a in "zyx"],
               "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}]}]}],
               "source_tifxyz": os.path.abspath(tifxyz), "volume": vol_url, "level": level, "flip": flip},
              open(os.path.join(out, ".zattrs"), "w"))
    json.dump({"zarr_format": 2, "shape": [layers, H, W], "chunks": [layers, 128, 128], "dtype": "|u1",
               "compressor": None, "fill_value": 0, "order": "C", "filters": None, "dimension_separator": "/"},
              open(os.path.join(out, "0", ".zarray"), "w"))
    canvas = np.zeros((layers, H, W), np.uint8)
    offs = ((np.arange(layers * k) - (layers * k - 1) / 2)).astype(np.float32)
    tile = (tile // k) * k
    pool = ThreadPoolExecutor(32)
    tiles = [(y, x) for y in range(0, Hf, tile) for x in range(0, Wf, tile) if vm[y:y + tile, x:x + tile].any()]
    for i, (y, x) in enumerate(tiles):
        m = vm[y:y + tile, x:x + tile]
        p = pts[y:y + tile, x:x + tile][m] / sc  # (N,3) xyz
        nn = n[y:y + tile, x:x + tile][m]
        q = p[None] + offs[:, None, None] * nn[None]  # (L,N,3) xyz, layer step = 1 level voxel
        zyx = q[..., ::-1]
        lo = np.floor(zyx.reshape(-1, 3).min(0)) - 1
        hi = np.ceil(zyx.reshape(-1, 3).max(0)) + 2
        crop, o = vol.read(lo, hi, pool)
        if crop.size == 0:
            continue
        c = (zyx - np.array(o)).reshape(-1, 3).T
        v = map_coordinates(crop, c, order=1, mode="constant", cval=0).reshape(layers * k, -1)
        fine = np.zeros((layers * k,) + m.shape, np.float32)
        fine[:, m] = v
        th, tw = m.shape[0] // k, m.shape[1] // k
        fine = fine[:, :th * k, :tw * k].reshape(layers, k, th, k, tw, k).mean((1, 3, 5))
        ok = m[:th * k, :tw * k].reshape(th, k, tw, k).all((1, 3))
        blk = canvas[:, y // k:y // k + th, x // k:x // k + tw]
        blk[:, ok] = np.clip(np.rint(fine[:, ok]), 0, 255).astype(np.uint8)
        if i % 20 == 0:
            print(f"\r tile {i + 1}/{len(tiles)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    for cy in range(0, (H + 127) // 128):
        for cx in range(0, (W + 127) // 128):
            b = np.zeros((layers, 128, 128), np.uint8)
            s = canvas[:, cy * 128:(cy + 1) * 128, cx * 128:(cx + 1) * 128]
            if not s.any():
                continue
            b[:, :s.shape[1], :s.shape[2]] = s
            d = os.path.join(out, "0", "0", str(cy))
            os.makedirs(d, exist_ok=True)
            b.tofile(os.path.join(d, str(cx)))
    return canvas, vm


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tifxyz"); ap.add_argument("volume"); ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=31); ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--pool", type=int, default=1, help="render k-times finer, then mean-pool by k")
    ap.add_argument("--frame-level", type=int, default=0, help="pyramid level the mesh coordinates live in")
    a = ap.parse_args()
    cv, vm = render(a.tifxyz, a.volume, a.out, a.layers, a.level, flip=a.flip, frame_level=a.frame_level, pool_k=a.pool)
    mid = cv[a.layers // 2 - 6:a.layers // 2 + 7].max(0)
    cv2.imwrite(a.out.rstrip("/") + "_max.png", mid)
