"""Rot90/transpose TTA helpers (used for the 8-way TTA row in FT9_RESULTS.md) for the RESULTS9 held-out inputs.

    python tta_transpose.py make            # tta/<case>/<input>_T.zarr = input with Y and X swapped
    python tta_transpose.py back <stem>     # tta/<case>/<input>_T_<stem>.tif -> tta/<case>/<input>_<stem>T.tif

Mirror TTA on the transposed input covers the other four elements of the square's symmetry group,
so <stem> together with <stem>T is the full 8-way dihedral TTA.
"""
import os, sys
import numpy as np, tifffile, zarr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prepare_inputs import write_zarr

CASES = ["pherc0139-w016", "pherc0814-46527"]
INPUTS = ["mac9", "off9"]

if sys.argv[1] == "make":
    for c in CASES:
        for inp in INPUTS:
            out = os.path.join(HERE, "tta", c, f"{inp}_T.zarr")
            if os.path.exists(out):
                continue
            vol = np.asarray(zarr.open_array(os.path.join(HERE, "data", c, f"{inp}.zarr", "0"), mode="r")[:])
            write_zarr(out, np.ascontiguousarray(vol.transpose(0, 2, 1)))
            print(c, inp, vol.shape)
else:
    stem = sys.argv[2]
    for c in CASES:
        for inp in INPUTS:
            p = tifffile.imread(os.path.join(HERE, "tta", c, f"{inp}_T_{stem}.tif"))
            tifffile.imwrite(os.path.join(HERE, "tta", c, f"{inp}_{stem}T.tif"), np.ascontiguousarray(p.T))
