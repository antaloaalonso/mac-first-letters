"""Render a tifxyz surface into a flat surface-volume zarr, streaming the scroll volume from the
open-data bucket (uncompressed 128^3 zarr v2 chunks). Port of ../render.py (validated r=0.968
against an official surface volume): same geometry, layer convention and output layout, but
  - every chunk a winding needs is prefetched in parallel before sampling,
  - trilinear sampling runs on the GPU (torch grid_sample, align_corners=True == map_coordinates order=1).
"""
import json, os, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np, tifffile, cv2

CACHE = os.environ.get("CHUNK_CACHE", "/kaggle/tmp/chunks")


class RemoteVolume:
    def __init__(self, url, level=0, cache=None):
        self.url = f"{url.rstrip('/')}/{level}"
        meta = json.loads(urllib.request.urlopen(self.url + "/.zarray", timeout=60).read())
        assert meta["compressor"] is None and meta["dtype"] == "|u1", meta
        self.shape, self.cs = meta["shape"], meta["chunks"][0]
        key = url.rstrip("/").split("/")
        self.cache = os.path.join(cache or CACHE, key[-3], key[-1], str(level))

    def path(self, cz, cy, cx):
        return os.path.join(self.cache, f"{cz}/{cy}/{cx}")

    def fetch(self, cz, cy, cx):
        p = self.path(cz, cy, cx)
        if os.path.exists(p):
            return
        raw = b""
        for attempt in range(8):
            try:
                raw = urllib.request.urlopen(f"{self.url}/{cz}/{cy}/{cx}", timeout=60).read()
                break
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):
                    break  # masked / absent chunk = fill 0
                if attempt == 7:
                    raise
            except Exception:
                if attempt == 7:
                    raise
            time.sleep(1 + attempt)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.{threading.get_ident()}.part"
        open(tmp, "wb").write(raw)
        os.replace(tmp, p)

    def chunk(self, cz, cy, cx):
        self.fetch(cz, cy, cx)
        raw = open(self.path(cz, cy, cx), "rb").read()
        c = self.cs
        if not raw:
            return None
        return np.frombuffer(raw, np.uint8).reshape(c, c, c)

    def keys(self, lo, hi):
        lo = [max(0, int(v)) for v in lo]
        hi = [min(s, int(v)) for v, s in zip(hi, self.shape)]
        if any(h <= l for l, h in zip(lo, hi)):
            return []
        c = self.cs
        return [(z, y, x) for z in range(lo[0] // c, (hi[0] - 1) // c + 1)
                for y in range(lo[1] // c, (hi[1] - 1) // c + 1)
                for x in range(lo[2] // c, (hi[2] - 1) // c + 1)]

    def read(self, lo, hi, pool):
        """Dense crop [lo, hi) in (z, y, x) voxels, zeros outside."""
        lo = [max(0, int(v)) for v in lo]
        hi = [min(s, int(v)) for v, s in zip(hi, self.shape)]
        out = np.zeros([max(0, h - l) for l, h in zip(lo, hi)], np.uint8)
        if out.size == 0:
            return out, lo
        c = self.cs
        idx = self.keys(lo, hi)
        for (z, y, x), ch in zip(idx, pool.map(lambda t: self.chunk(*t), idx)):
            if ch is None:
                continue
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
    # output pixel j <-> mesh coordinate j / up (the published surface volumes'
    # grid); cv2.resize's pixel-centre convention would shift by (up - 1) / 2 px
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


def sample_trilinear(crop, zyx, device):
    """crop (D,H,W) uint8, zyx (M,3) float voxel coords into crop -> (M,) float32, 0 outside."""
    import torch
    D, H, W = crop.shape
    vol = torch.from_numpy(np.ascontiguousarray(crop)).to(device, non_blocking=True).float()[None, None]
    g = torch.from_numpy(np.ascontiguousarray(zyx, np.float32)).to(device)
    size = torch.tensor([W - 1, H - 1, D - 1], device=device, dtype=torch.float32).clamp_min(1)
    grid = (g[:, [2, 1, 0]] / size * 2 - 1).view(1, 1, 1, -1, 3)
    out = torch.nn.functional.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out.view(-1).cpu().numpy()


def render(tifxyz, vol_url, out, layers=31, level=0, tile=256, flip=False, frame_level=0,
           device="cpu", threads=64, log=print):
    """Returns (canvas [L,H,W] uint8, valid mask). Writes an uncompressed zarr v2 at `out`."""
    sc = 2 ** (level - frame_level)
    meta = json.load(open(os.path.join(tifxyz, "meta.json")))
    pts, vm = load_tifxyz(tifxyz, step=max(1, round(1 / meta["scale"][0] / sc)))
    n = normals(pts) * (-1 if flip else 1)
    vol = RemoteVolume(vol_url, level)
    H, W = vm.shape
    half = layers // 2
    offs = (np.arange(layers) - (layers - 1) / 2).astype(np.float32)
    tiles = []
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            m = vm[y:y + tile, x:x + tile]
            if not m.any():
                continue
            p = pts[y:y + tile, x:x + tile][m] / sc
            zyx_c = p[:, ::-1]
            ext = half + 2
            lo = np.floor(zyx_c.min(0)) - ext
            hi = np.ceil(zyx_c.max(0)) + ext + 1
            tiles.append((y, x, lo, hi))
    keys = sorted({k for _, _, lo, hi in tiles for k in vol.keys(lo, hi)})
    t0 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(lambda k: vol.fetch(*k), keys))
    log(f"   prefetched {len(keys)} chunks in {time.time() - t0:.0f}s; {len(tiles)} tiles {H}x{W}")
    canvas = np.zeros((layers, H, W), np.uint8)
    pool = ThreadPoolExecutor(16)
    for i, (y, x, _, _) in enumerate(tiles):
        m = vm[y:y + tile, x:x + tile]
        p = pts[y:y + tile, x:x + tile][m] / sc
        nn = n[y:y + tile, x:x + tile][m]
        q = p[None] + offs[:, None, None] * nn[None]
        zyx = q[..., ::-1].reshape(-1, 3)
        lo = np.floor(zyx.min(0)) - 1
        hi = np.ceil(zyx.max(0)) + 2
        crop, o = vol.read(lo, hi, pool)
        if crop.size == 0 or not crop.any():
            continue
        v = sample_trilinear(crop, zyx - np.array(o, np.float32), device).reshape(layers, -1)
        blk = canvas[:, y:y + m.shape[0], x:x + m.shape[1]]
        blk[:, m] = np.clip(np.rint(v), 0, 255).astype(np.uint8)
    pool.shutdown()
    write_zarr(out, canvas, {"source_tifxyz": os.path.abspath(tifxyz), "volume": vol_url, "level": level, "flip": flip})
    return canvas, vm


def write_zarr(out, canvas, attrs):
    layers, H, W = canvas.shape
    os.makedirs(os.path.join(out, "0"), exist_ok=True)
    json.dump({"zarr_format": 2}, open(os.path.join(out, ".zgroup"), "w"))
    json.dump({"multiscales": [{"version": "0.4", "axes": [{"name": a, "type": "space"} for a in "zyx"],
               "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}]}]}],
               **attrs}, open(os.path.join(out, ".zattrs"), "w"))
    json.dump({"zarr_format": 2, "shape": [layers, H, W], "chunks": [layers, 128, 128], "dtype": "|u1",
               "compressor": None, "fill_value": 0, "order": "C", "filters": None, "dimension_separator": "/"},
              open(os.path.join(out, "0", ".zarray"), "w"))
    for cy in range(0, (H + 127) // 128):
        for cx in range(0, (W + 127) // 128):
            s = canvas[:, cy * 128:(cy + 1) * 128, cx * 128:(cx + 1) * 128]
            if not s.any():
                continue
            b = np.zeros((layers, 128, 128), np.uint8)
            b[:, :s.shape[1], :s.shape[2]] = s
            d = os.path.join(out, "0", "0", str(cy))
            os.makedirs(d, exist_ok=True)
            b.tofile(os.path.join(d, str(cx)))


def prune_cache(root, max_gb):
    files = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            try:
                st = os.stat(p)
            except FileNotFoundError:
                continue
            files.append((st.st_mtime, st.st_size, p))
    tot = sum(s for _, s, _ in files)
    for _, s, p in sorted(files):
        if tot <= max_gb * 1e9:
            break
        os.remove(p); tot -= s
    return tot
