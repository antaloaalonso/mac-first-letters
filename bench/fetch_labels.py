"""Fetch the ink_9um validation region of one labelled segment.

    python fetch_labels.py <case> <out_dir>      e.g. pherc0139-w016 bench/data

Lists the chunks of <case>_validation_mask.zarr in the scrollprize/datasets
bucket, takes their bounding box (in level-2 pixels of the 2.4 um surface
volume), and writes the annotated plane (z=10) of the ink labels, validation
mask and supervision mask inside it to <out_dir>/<case>/labels.npz together
with the box.
"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import numcodecs

API = "https://huggingface.co/api/buckets/scrollprize/datasets/tree/"
RAW = "https://huggingface.co/buckets/scrollprize/datasets/resolve/"
ROOT = "ink_9um/labels/aligned-scrollprizeorg-21slices"


def fetch(url):
    """GET with backoff: the bucket rate-limits anonymous clients (HTTP 429)."""
    for attempt in range(8):
        try:
            return urllib.request.urlopen(url, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 7:
                raise
            time.sleep(min(120, 5 * 2 ** attempt))


def list_chunks(prefix):
    url = f"{API}{prefix}?recursive=true&limit=1000"
    keys = []
    while url:
        with fetch(url) as r:
            keys += [e["path"] for e in json.load(r) if e["type"] == "file"]
            link = r.headers.get("Link") or ""
        url = link.split(";")[0].strip("<> ") if 'rel="next"' in link else None
    return keys


def read_plane(prefix, chunk_ids, meta, plane, box):
    y0, y1, x0, x1 = box
    out = np.zeros((y1 - y0, x1 - x0), np.uint8)
    codec = numcodecs.get_codec(meta["compressor"])
    cz, cy, cx = meta["chunks"]

    def get(cid):
        _, iy, ix = cid
        raw = fetch(f"{RAW}{prefix}/0/0.{iy}.{ix}").read()
        return cid, np.frombuffer(codec.decode(raw), np.uint8).reshape(cz, cy, cx)[plane]

    with ThreadPoolExecutor(4) as pool:
        for (_, iy, ix), a in pool.map(get, chunk_ids):
            if a is None:
                continue
            ys, xs = iy * cy, ix * cx
            sy0, sx0 = max(ys, y0), max(xs, x0)
            sy1, sx1 = min(ys + cy, y1), min(xs + cx, x1)
            out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0 - ys:sy1 - ys, sx0 - xs:sx1 - xs]
    return out


def _chunk_ids(keys):
    return sorted({tuple(int(v) for v in k.rsplit("/", 1)[1].split(".")) for k in keys
                   if k.rsplit("/", 1)[1].count(".") == 2})


def _present(prefix, box, cy, cx):
    """Existing chunks of an array that overlap the box (absent chunks are all zero)."""
    y0, y1, x0, x1 = box
    return [i for i in _chunk_ids(list_chunks(prefix))
            if y0 // cy <= i[1] < -(-y1 // cy) and x0 // cx <= i[2] < -(-x1 // cx)]


def tighten(planes, box):
    ys, xs = np.nonzero(planes["validation_mask"])
    ty0, ty1, tx0, tx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    planes = {k: v[ty0:ty1, tx0:tx1] for k, v in planes.items()}
    return planes, (box[0] + int(ty0), box[0] + int(ty1), box[2] + int(tx0), box[2] + int(tx1))


def main(case, out_dir):
    base = f"{ROOT}/{case}/{case}"
    meta = json.load(fetch(f"{RAW}{base}_validation_mask.zarr/0/.zarray"))
    attrs = json.load(fetch(f"{RAW}{base}_validation_mask.zarr/.zattrs"))
    plane = int(attrs.get("annotation_center_channel", 10))
    cy, cx = meta["chunks"][1:]
    keys = list_chunks(f"{base}_validation_mask.zarr/0")
    ids = _chunk_ids(keys)
    iy = [i[1] for i in ids]; ix = [i[2] for i in ids]
    H, W = meta["shape"][1:]
    box = (min(iy) * cy, min((max(iy) + 1) * cy, H), min(ix) * cx, min((max(ix) + 1) * cx, W))
    planes = {}
    for k in ("validation_mask", "inklabels", "supervision_mask"):
        present = ids if k == "validation_mask" else _present(f"{base}_{k}.zarr/0", box, cy, cx)
        planes[k] = read_plane(f"{base}_{k}.zarr", present, meta, plane, box)
    # The bucket stores all-zero chunks too, so tighten to the masked pixels.
    planes, box = tighten(planes, box)
    os.makedirs(os.path.join(out_dir, case), exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, case, "labels.npz"), box=np.array(box), shape=np.array([H, W]), **planes)
    v = planes["validation_mask"] > 0
    print(f"{case}: box y[{box[0]},{box[1]}) x[{box[2]},{box[3]}) of {H}x{W}; "
          f"validation px {int(v.sum())}, ink px in validation {int((planes['inklabels'] > 0)[v].sum())}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
