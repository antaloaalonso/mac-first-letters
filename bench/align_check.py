"""Compare the centre layers of official.zarr and mac.zarr for one case.

    python align_check.py <data_dir> <case>

Prints the normalised cross-correlation at zero shift and the best integer
(dy, dx) shift within +-8 px, over pixels valid in both inputs.
"""
import json, os, sys
import numpy as np, zarr


def ncc(a, b):
    m = (a > 0) & (b > 0)
    if m.sum() < 1000:
        return float("nan")
    x, y = a[m].astype(np.float64), b[m].astype(np.float64)
    x -= x.mean(); y -= y.mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def main(data_dir, case):
    cdir = os.path.join(data_dir, case)
    off = np.asarray(zarr.open(os.path.join(cdir, "official.zarr", "0"), mode="r"))
    mac = np.asarray(zarr.open(os.path.join(cdir, "mac.zarr", "0"), mode="r"))
    a, b = off[10].astype(np.float32), mac[10].astype(np.float32)
    best = max(((ncc(a[8 + dy:a.shape[0] - 8 + dy, 8 + dx:a.shape[1] - 8 + dx], b[8:-8, 8:-8]), dy, dx)
                for dy in range(-8, 9) for dx in range(-8, 9)), key=lambda t: -np.inf if np.isnan(t[0]) else t[0])
    layers = [ncc(off[z], mac[z]) for z in range(off.shape[0])]
    out = {"case": case, "ncc_centre_zero_shift": ncc(a, b), "best_ncc": best[0], "best_shift_dy_dx": best[1:],
           "ncc_per_layer_min": float(np.nanmin(layers)), "ncc_per_layer_max": float(np.nanmax(layers)),
           "best_layer": int(np.nanargmax(layers))}
    print(json.dumps(out))
    json.dump(out, open(os.path.join(cdir, "align.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
