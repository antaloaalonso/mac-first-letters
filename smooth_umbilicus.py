"""Median-smooth an umbilicus written by vc_gen_umbilicus, in place.

    python smooth_umbilicus.py <umbilicus.json> [window]

Replaces each control point's x and y by the median over a window of
neighbouring points along z (default 5). Single-slice estimates jump by
hundreds of voxels where the centre is ambiguous; the fit wants a smooth axis.
"""
import json, sys
import numpy as np

path = sys.argv[1]
window = int(sys.argv[2]) if len(sys.argv) > 2 else 5
doc = json.load(open(path))
pts = sorted(doc["control_points"], key=lambda p: p["z"])
xy = np.array([[p["x"], p["y"]] for p in pts], float)
h = window // 2
smooth = np.array([np.median(xy[max(0, i - h):i + h + 1], axis=0) for i in range(len(xy))])
for p, (x, y) in zip(pts, smooth):
    p["x"], p["y"] = float(x), float(y)
doc["control_points"] = pts
doc.setdefault("metadata", {})["smoothing"] = f"median over {window} control points"
json.dump(doc, open(path, "w"), indent=1)
print(f"{path}: {len(pts)} points, max shift {np.abs(smooth - xy).max():.1f} voxels")
