"""Training data for adapting ink_9um to 9.36 um input: the ink_9um `native9` labels.

    python fetch_native9.py <case> <out_dir>        case in w035 w039 w040 w041 w044

The native9 labels (PHerc0139, five segments) are annotated directly on the published 9.362 um
surface volumes, so they need no registration. None of these segments is one of the held-out
validation segments (pherc0139-w016 = public w029, pherc0814-46527). Writes <out_dir>/<case>/:
  vol.npy     uint8 (28, h, w): the published 9.362 um surface volume over the labelled box + margin
  labels.npz  inklabels / supervision_mask (annotated plane, same h, w) and the box
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fetch_labels as fl
from res9 import read_crop
from prepare_inputs import S3

ROOT9 = "ink_9um/labels/native9-scrollprizeorg-21slices"
SEGMENTS = {
    "w035": "20260317000000-w035_2026031718", "w039": "20260302000000-w039_2026030210",
    "w040": "20250831000000-w040_2025083102", "w041": "20260108000000-w041_2026010816",
    "w044": "20260115000000-w044_2026011522",
}
VOL9 = "9.362um-1.2m-113keV-volume-20250728140407.zarr"
MARGIN = 64


def main(case, out_dir):
    base = f"{ROOT9}/{case}/{case}"
    meta = json.load(fl.fetch(f"{fl.RAW}{base}_supervision_mask.zarr/0/.zarray"))
    attrs = json.load(fl.fetch(f"{fl.RAW}{base}_inklabels.zarr/.zattrs"))
    plane = int(attrs.get("annotation_center_channel", 14))
    _, H, W = meta["shape"]; cy, cx = meta["chunks"][1:]
    sup_ids = fl._chunk_ids(fl.list_chunks(f"{base}_supervision_mask.zarr/0"))
    iy = [i[1] for i in sup_ids]; ix = [i[2] for i in sup_ids]
    box = (min(iy) * cy, min((max(iy) + 1) * cy, H), min(ix) * cx, min((max(ix) + 1) * cx, W))
    planes = {"supervision_mask": fl.read_plane(f"{base}_supervision_mask.zarr", sup_ids, meta, plane, box)}
    planes["inklabels"] = fl.read_plane(f"{base}_inklabels.zarr", fl._present(f"{base}_inklabels.zarr/0", box, cy, cx),
                                        meta, plane, box)
    ys, xs = np.nonzero(planes["supervision_mask"])
    t = (max(0, box[0] + ys.min() - MARGIN), min(H, box[0] + ys.max() + 1 + MARGIN),
         max(0, box[2] + xs.min() - MARGIN), min(W, box[2] + xs.max() + 1 + MARGIN))
    planes = {k: v[t[0] - box[0]:t[1] - box[0], t[2] - box[2]:t[3] - box[2]] for k, v in planes.items()}
    vol, got = read_crop(f"{S3}/PHerc0139/segments/{SEGMENTS[case]}/surface-volumes/{VOL9}/0", t)
    assert tuple(got) == tuple(t), (got, t)
    d = os.path.join(out_dir, case); os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "vol.npy"), vol)
    np.savez_compressed(os.path.join(d, "labels.npz"), box=np.array(t), **planes)
    sup = planes["supervision_mask"] > 0
    print(f"{case}: box {t}, vol {vol.shape}, supervised px {int(sup.sum())}, "
          f"ink px {int((planes['inklabels'] > 0)[sup].sum())} ({(planes['inklabels'] > 0)[sup].mean():.2f})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
