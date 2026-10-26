"""Stage a spiral-fit dataset for one z band without bulk downloads.

- .vctracks index arrays: whole (a few hundred MB); coordinates.i32: sparse file holding only the
  byte ranges of tracks lying wholly inside [z0, z1) (the fit's packed-store loader keeps exactly
  those tracks, so the rest is never read);
- the .dbm SQLite: a sparse stub (real 4 KB header, original size and mtime) so the packed store's
  and the crossings cache's `source_db_signature` match without the 5-12 GB file;
- crossings .npz: whole;
- lasagna nx/ny/grad_mag: only the z chunk rows of the band, at the 4x group;
- umbilicus: published one if present, else estimated (lasagna normals, else CT mask centroid).
"""
import json, os, re, time, urllib.request, urllib.error, concurrent.futures as cf
import numpy as np

DL = "https://dl.ash2txt.org/datasets/spiral_datasets"
S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com"


def http(url, rng=None, timeout=120, tries=8):
    err = None
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1] - 1}"} if rng else {})
            return urllib.request.urlopen(req, timeout=timeout).read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            err = e
        except Exception as e:
            err = e
        time.sleep(min(30, 2 * (a + 1)))
    raise RuntimeError(f"{url} {rng}: {err}")


def head_size(url):
    req = urllib.request.Request(url, method="HEAD")
    return int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])


def download(url, path, threads=16, part=64 << 20):
    """Parallel ranged download of one big file (resumable per part)."""
    if os.path.exists(path):
        return path
    size = head_size(url)
    tmp = path + ".part"
    with open(tmp, "ab") as f:
        f.truncate(size)
    done_p = tmp + ".done"
    done = set(json.load(open(done_p))) if os.path.exists(done_p) else set()
    fd = os.open(tmp, os.O_WRONLY)
    parts = [(s, min(size, s + part)) for s in range(0, size, part) if s not in done]

    def get(r):
        b = http(url, r)
        assert b is not None and len(b) == r[1] - r[0], f"short read {r}"
        os.pwrite(fd, b, r[0])
        return r[0]
    with cf.ThreadPoolExecutor(threads) as ex:
        for i, s in enumerate(ex.map(get, parts)):
            done.add(s)
            if i % 8 == 0:
                json.dump(sorted(done), open(done_p, "w"))
                print(f"  {os.path.basename(path)} {len(done) * part / 1e9:.1f}/{size / 1e9:.1f} GB", flush=True)
    os.close(fd)
    os.replace(tmp, path)
    if os.path.exists(done_p):
        os.remove(done_p)
    return path


def s3_list(prefix):
    out_p, out_k, token = [], [], None
    while True:
        u = f"{S3}/?list-type=2&delimiter=/&prefix={prefix}" + (f"&continuation-token={urllib.request.quote(token)}" if token else "")
        h = http(u).decode()
        out_p += re.findall(r"<Prefix>([^<]+)</Prefix>", h)[1:]
        out_k += re.findall(r"<Key>([^<]+)</Key>", h)
        m = re.search(r"<NextContinuationToken>([^<]+)<", h)
        if not m:
            return out_p, out_k
        token = m.group(1)


def resolve_volume(scroll, volume):
    """'20250821151701' or a full name -> (full zarr name, voxel size um)."""
    vols = [p.rstrip("/").split("/")[-1] for p in s3_list(f"{scroll}/volumes/")[0]]
    m = [v for v in vols if v.startswith(str(volume))]
    if not m:
        raise SystemExit(f"volume {volume} not in {vols}")
    return m[0], float(re.search(r"-([\d.]+)um-", m[0]).group(1))


def stage_tracks(scroll, vol_ts, ds):
    base = f"{DL}/{scroll}/{vol_ts}/tracks/"
    names = re.findall(r'href="([^"?]+)"', http(base).decode())
    dbm = next(n for n in names if n.endswith(".dbm"))
    crossings = dbm + ".crossings.npz"
    store = os.path.join(ds, "tracks", dbm + ".vctracks")
    os.makedirs(store, exist_ok=True)
    for f in ("header.bin", "metadata.json", "z_bounds.i32", "offsets.i64", "source_ids.u64",
              "arclengths.f64", "tortuosities.f64", "family_codes.i8"):
        download(f"{base}{dbm}.vctracks/{f}", os.path.join(store, f), threads=8, part=16 << 20)
    meta = json.load(open(os.path.join(store, "metadata.json")))
    _, size, mtime_ns = meta["source_db_signature"][0]
    stub = os.path.join(ds, "tracks", dbm)
    if not os.path.exists(stub):
        hdr = http(base + dbm, (0, 4096))
        with open(stub, "wb") as f:
            f.write(hdr); f.truncate(size)
        os.utime(stub, ns=(mtime_ns, mtime_ns))
    download(base + crossings, os.path.join(ds, "tracks", crossings), threads=16)
    return store, dbm


def auto_band(store, height=1000, step=250):
    zb = np.fromfile(f"{store}/z_bounds.i32", "<i4").reshape(-1, 2)
    off = np.fromfile(f"{store}/offsets.i64", "<i8")
    n = np.diff(off)
    best = None
    for z0 in range(int(zb[:, 0].min()) // step * step, int(zb[:, 1].max()) - height, step):
        s = int(n[(zb[:, 0] >= z0) & (zb[:, 1] < z0 + height)].sum())
        if best is None or s > best[1]:
            best = (z0, s)
    return best[0], best[0] + height, best[1]


def stage_band_coords(store, dbm_url, z0, z1, gap=512 * 1024, threads=24):
    """Sparse coordinates.i32 with only the band's tracks (resumable: range list is recorded)."""
    url = dbm_url + ".vctracks/coordinates.i32"
    zb = np.fromfile(f"{store}/z_bounds.i32", "<i4").reshape(-1, 2)
    off = np.fromfile(f"{store}/offsets.i64", "<i8")
    idx = np.nonzero((zb[:, 0] >= z0) & (zb[:, 1] < z1))[0]
    ranges = []
    for s, e in zip(off[idx] * 12, off[idx + 1] * 12):
        if ranges and s - ranges[-1][1] <= gap:
            ranges[-1][1] = max(ranges[-1][1], int(e))
        else:
            ranges.append([int(s), int(e)])
    path = f"{store}/coordinates.i32"
    mark = f"{path}.band_{z0}_{z1}"
    if os.path.exists(mark):
        return len(idx)
    print(f"  band [{z0},{z1}): {len(idx)} tracks, {len(ranges)} ranges, "
          f"{sum(e - s for s, e in ranges) / 1e9:.2f} GB", flush=True)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.truncate(int(off[-1]) * 12)
    fd = os.open(path, os.O_WRONLY)

    def get(r):
        b = http(url, tuple(r))
        assert b is not None and len(b) == r[1] - r[0]
        os.pwrite(fd, b, r[0])
        return len(b)
    tot = 0
    with cf.ThreadPoolExecutor(threads) as ex:
        for i, n in enumerate(ex.map(get, ranges)):
            tot += n
            if i % 1000 == 0:
                print(f"  {i}/{len(ranges)} {tot / 1e9:.2f} GB", flush=True)
    os.close(fd)
    open(mark, "w").write("ok")
    return len(idx)


def stage_lasagna(scroll, vol_ts, ds, z0, z1, want_scale=4):
    """Mirror the band's z rows of nx/ny/grad_mag. Returns (group, scale, stores) or None."""
    pre = f"{scroll}/representations/predictions/lasagna/"
    runs = [p for p in s3_list(pre)[0] if p.rstrip("/").split("/")[-1].startswith(vol_ts)]
    if not runs:
        return None
    run = sorted(runs)[-1]
    stores = {}
    for p in s3_list(run)[0]:
        name = p.rstrip("/").split("/")[-1]
        for k in ("nx", "ny", "grad_mag"):
            if name.endswith(f"_{k}.ome.zarr"):
                stores[k] = (p, name)
    if set(stores) != {"nx", "ny", "grad_mag"}:
        return None
    za = json.loads(http(f"{S3}/{stores['nx'][0]}.zattrs"))
    group, scale = None, None
    for d in za["multiscales"][0]["datasets"]:
        sc = d["coordinateTransformations"][0]["scale"][0]
        if scale is None or abs(sc - want_scale) < abs(scale - want_scale):
            group, scale = d["path"], int(round(sc))
    out_dir = os.path.join(ds, "lasagna_inputs")
    jobs = []
    for k, (p, name) in stores.items():
        out = os.path.join(out_dir, name)
        for rel in (".zattrs", ".zgroup", f"{group}/.zarray", f"{group}/.zattrs"):
            fp = os.path.join(out, rel)
            if not os.path.exists(fp):
                b = http(f"{S3}/{p}{rel}")
                if b is not None:
                    os.makedirs(os.path.dirname(fp), exist_ok=True); open(fp, "wb").write(b)
        a = json.load(open(os.path.join(out, group, ".zarray")))
        cz, cy, cx = a["chunks"]; Z, Y, X = a["shape"]
        zr = range(max(0, (z0 // scale) // cz - 1), min(-(-(z1 // scale) // cz) + 1, -(-Z // cz)))
        jobs += [(f"{S3}/{p}{group}/{i}/{j}/{l}", os.path.join(out, group, str(i), str(j), str(l)))
                 for i in zr for j in range(-(-Y // cy)) for l in range(-(-X // cx))]

    def fetch(job):
        u, fp = job
        if os.path.exists(fp):
            return 1
        b = http(u, timeout=60)
        if b is None:
            return 0
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        open(fp + ".tmp", "wb").write(b); os.replace(fp + ".tmp", fp)
        return 1
    print(f"  lasagna group {group} (x{scale}): {len(jobs)} chunks", flush=True)
    n = 0
    with cf.ThreadPoolExecutor(64) as ex:
        for i, r in enumerate(ex.map(fetch, jobs)):
            n += r
            if i % 20000 == 0:
                print(f"  {i}/{len(jobs)} ({n} present)", flush=True)
    return group, scale, {k: os.path.join(out_dir, v[1]) for k, v in stores.items()}


def published_umbilicus(scroll, vol_ts):
    keys = [k for k in s3_list(f"{scroll}/representations/umbilicus/")[1]
            if k.split("/")[-1].startswith(vol_ts) and k.endswith(".json")]
    if not keys:
        return None
    return json.loads(http(f"{S3}/{sorted(keys)[-1]}"))


def lasagna_umbilicus(stores, group, scale, z0, z1):
    import zarr
    from umbilicus_est import estimate
    nx, ny, gm = [zarr.open_array(os.path.join(stores[k], group), mode="r") for k in ("nx", "ny", "grad_mag")]
    zs = list(range(z0 // scale, z1 // scale, max(1, 64 // scale)))
    # lasagna nx/ny: the stored "nx" behaves as the row (y) component in the zarr's (z, y, x) grid;
    # (ny, nx) put the estimate within ~500 px of PHerc0826's published umbilicus (crushed scroll),
    # (nx, ny) was ~2000 px off.
    return estimate(lambda z: (ny[z], nx[z], gm[z]), zs, scale)


def ct_umbilicus(vol_url, z0, z1, level=3):
    """Fallback: intensity-mask centroid of a few low-res CT slices."""
    from render_gpu import RemoteVolume
    v = RemoteVolume(vol_url, level)
    s = 2 ** level
    pts = []
    pool = cf.ThreadPoolExecutor(32)
    for z in np.linspace(z0, z1 - 1, 5):
        zl = int(z) // s
        sl, _ = v.read((zl, 0, 0), (zl + 1, v.shape[1], v.shape[2]), pool)
        a = sl[0].astype(np.float32)
        if not (a > 0).any():
            continue
        m = a > np.percentile(a[a > 0], 60)
        ys, xs = np.nonzero(m)
        pts.append({"x": float(xs.mean() * s), "y": float(ys.mean() * s), "z": float(z)})
    return pts


def write_spec(ds, scroll, voxel, sense, dbm, lasagna):
    spec = {"schema_version": 1, "name": scroll, "voxel_size_um": voxel, "spiral_outward_sense": sense,
            "paths": {"tracks_dbm": f"tracks/{dbm}"}}
    if lasagna:
        group, scale, stores = lasagna
        spec.update(normal_zarr_group=group, lasagna_scale=scale)
        spec["paths"].update(normal_x=os.path.relpath(stores["nx"], ds), normal_y=os.path.relpath(stores["ny"], ds),
                             gradient_magnitude=os.path.relpath(stores["grad_mag"], ds))
    json.dump(spec, open(os.path.join(ds, "spiral-scroll.json"), "w"), indent=1)
    return spec
