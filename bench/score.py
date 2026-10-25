"""Score ink predictions against the held-out ink_9um labels.

    python score.py <data_dir> <case> <pred.tif> [<pred.tif> ...]

Each prediction must cover the case's input box (boxes.json). Pixels inside
the validation mask are scored against the ink labels (label > 0). Reports
AUROC, average precision, and precision/recall/F1 at the best-F1 threshold,
one JSON line per prediction.
"""
import json, os, sys
import numpy as np, tifffile


def roc_pr(scores, labels):
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order].astype(np.int64)
    distinct = np.r_[np.nonzero(np.diff(s))[0], y.size - 1]
    tp = np.cumsum(y)[distinct]
    fp = (distinct + 1) - tp
    P, N = int(y.sum()), int(y.size - y.sum())
    tpr, fpr = tp / P, fp / N
    auroc = float(np.trapezoid(np.r_[0, tpr], np.r_[0, fpr]))
    precision = tp / (tp + fp)
    recall = tpr
    ap = float(np.sum(np.diff(np.r_[0, recall]) * precision))
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    i = int(np.argmax(f1))
    return {"auroc": auroc, "ap": ap, "best_f1": float(f1[i]), "precision": float(precision[i]),
            "recall": float(recall[i]), "threshold": float(s[distinct[i]]), "pos_frac": P / y.size}


def main(data_dir, case, preds):
    cdir = os.path.join(data_dir, case)
    lab = np.load(os.path.join(cdir, "labels.npz"))
    boxes = json.load(open(os.path.join(cdir, "boxes.json")))
    iy0, _, ix0, _ = boxes["input_box"]
    ly0, ly1, lx0, lx1 = boxes["label_box"]
    valid = lab["validation_mask"] > 0
    ink = lab["inklabels"] > 0
    for p in preds:
        pred = tifffile.imread(p).astype(np.float32)
        crop = pred[ly0 - iy0:ly1 - iy0, lx0 - ix0:lx1 - ix0]
        m = valid & np.isfinite(crop)
        res = roc_pr(crop[m], ink[m])
        print(json.dumps({"case": case, "pred": os.path.basename(p), "pixels": int(m.sum()), **res}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
