"""Score every case's predictions and write RESULTS.md / results.json.

    python report.py <data_dir> <out_dir>
"""
import json, os, subprocess, sys
import numpy as np, tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = ("official", "mac")
MODELS = (("seed42", "seed42"), ("seed43", "seed43"), ("ens", "seed42+43"))


def pred_agreement(cdir):
    """Pearson r between the official- and mac-input ensemble predictions inside the validation mask."""
    lab = np.load(os.path.join(cdir, "labels.npz"))
    boxes = json.load(open(os.path.join(cdir, "boxes.json")))
    iy0, _, ix0, _ = boxes["input_box"]; ly0, ly1, lx0, lx1 = boxes["label_box"]
    v = lab["validation_mask"] > 0
    a, b = [tifffile.imread(os.path.join(cdir, "pred", f"{i}_ens.tif")).astype(np.float64)
            [ly0 - iy0:ly1 - iy0, lx0 - ix0:lx1 - ix0][v] for i in INPUTS]
    return float(np.corrcoef(a, b)[0, 1])


def main(data_dir, out_dir):
    cases = sorted(c for c in os.listdir(data_dir) if os.path.exists(os.path.join(data_dir, c, "pred")))
    rows, extra = [], {}
    for c in cases:
        preds = [os.path.join(data_dir, c, "pred", f"{i}_{m}.tif") for i in INPUTS for m, _ in MODELS]
        out = subprocess.run([sys.executable, os.path.join(HERE, "score.py"), data_dir, c, *preds],
                             check=True, capture_output=True, text=True).stdout
        rows += [json.loads(l) for l in out.splitlines() if l.strip()]
        align = json.load(open(os.path.join(data_dir, c, "align.json")))
        extra[c] = {"input_ncc": align["ncc_centre_zero_shift"], "pred_r": pred_agreement(os.path.join(data_dir, c))}
    json.dump({"rows": rows, "per_case": extra}, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    get = {(r["case"], r["pred"]): r for r in rows}
    L = ["# Held-out ink benchmark: Mac streaming input vs. official surface volume", "",
         "ink_9um step-075000 on the three published validation regions (pixels inside "
         "`<case>_validation_mask`, scored against `<case>_inklabels`). `official` = level 2 of the "
         "published 2.4 µm surface volume pooled to 21 planes as villa's `prepare_9um_isotropic_input`; "
         "`mac` = the published tifxyz re-rendered by `render.py`, streaming only the CT chunks it touches.", "",
         "| case | val px | ink % | input | model | AUROC | AP | best F1 | P | R |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cases:
        for i in INPUTS:
            for m, name in MODELS:
                r = get[(c, f"{i}_{m}.tif")]
                L.append(f"| {c} | {r['pixels']:,} | {100 * r['pos_frac']:.0f} | {i} | {name} | {r['auroc']:.3f} | "
                         f"{r['ap']:.3f} | {r['best_f1']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} |")
    L += ["", "**Mean over cases**", "", "| input | model | AUROC | AP | best F1 |", "|---|---|---|---|---|"]
    for i in INPUTS:
        for m, name in MODELS:
            rs = [get[(c, f"{i}_{m}.tif")] for c in cases]
            L.append(f"| {i} | {name} | " + " | ".join(f"{np.mean([r[k] for r in rs]):.3f}"
                                                      for k in ("auroc", "ap", "best_f1")) + " |")
    L += ["", "**Input and prediction agreement** (inside the validation mask)", "",
          "| case | input NCC, centre layer | ensemble prediction r, official vs mac |", "|---|---|---|"]
    L += [f"| {c} | {extra[c]['input_ncc']:.3f} | {extra[c]['pred_r']:.3f} |" for c in cases]
    open(os.path.join(out_dir, "RESULTS.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
