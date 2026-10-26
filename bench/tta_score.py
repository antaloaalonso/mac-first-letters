"""Score (averages of) 9.36 um prediction TIFFs exactly like `res9.py score`.

    python tta_score.py <row name> <input> <tif stem> [<tif stem> ...]

Each stem is looked up per case in tta/<case>/ then data/<case>/pred9/ as <input>_<stem>.tif.
The probability maps are averaged (equal weights) before the RESULTS9 warp and scoring.
Appends one JSON line per call to tta/scores.jsonl and prints a table row.
"""
import json, os, sys
import numpy as np, tifffile, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score import roc_pr

CASES = ["pherc0139-w016", "pherc0814-46527"]


def load(case, inp, stem):
    for d in (os.path.join(HERE, "tta", case), os.path.join(HERE, "data", case, "pred9")):
        f = os.path.join(d, f"{inp}_{stem}.tif")
        if os.path.exists(f):
            return tifffile.imread(f).astype(np.float32)
    raise FileNotFoundError(f"{case} {inp}_{stem}.tif")


def score_case(case, p):
    cdir = os.path.join(HERE, "data", case)
    reg = json.load(open(os.path.join(cdir, "res9.json")))
    warp = np.array(reg["warp_9_to_label_input"], np.float32)
    boxes = json.load(open(os.path.join(cdir, "boxes.json")))
    iy0, iy1, ix0, ix1 = boxes["input_box"]; ly0, ly1, lx0, lx1 = boxes["label_box"]
    lab = np.load(os.path.join(cdir, "labels.npz"))
    valid, ink = lab["validation_mask"] > 0, lab["inklabels"] > 0
    w = cv2.warpAffine(p, warp, (ix1 - ix0, iy1 - iy0), flags=cv2.INTER_LINEAR, borderValue=np.nan)
    crop = w[ly0 - iy0:ly1 - iy0, lx0 - ix0:lx1 - ix0]
    m = valid & np.isfinite(crop)
    return roc_pr(crop[m], ink[m])


def main():
    name, inp, stems = sys.argv[1], sys.argv[2], sys.argv[3:]
    res = {}
    for c in CASES:
        p = np.mean([load(c, inp, s) for s in stems], axis=0)
        res[c] = score_case(c, p)
    a = [res[c]["auroc"] for c in CASES]; ap = [res[c]["ap"] for c in CASES]
    row = {"row": name, "input": inp, "stems": stems, **{c: res[c] for c in CASES},
           "mean_auroc": float(np.mean(a)), "mean_ap": float(np.mean(ap))}
    with open(os.path.join(HERE, "tta", "scores.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"| {name} | {inp} | {a[0]:.3f} / {ap[0]:.3f} | {a[1]:.3f} / {ap[1]:.3f} | {np.mean(a):.3f} | {np.mean(ap):.3f} |")


if __name__ == "__main__":
    main()
