"""The band sweep's tile score on held-out text: is a high |fwd - rev| a sign of letters at 9.36 um?

The band pipeline ranks 2136 px tiles by |area(fwd > 127) - area(rev > 127)| (kaggle/src/montage.py, after
void masking). This applies the same score to the two RESULTS9 held-out regions, which contain labelled
text, with the recommended screen (ft9_seed42 + base seed43, both --tta-mirror), and prints the best tile
per region. Needs the forward and reverse maps from tta_run.sh:

    tta_run.sh ft42_mir ft9/ckpt/ft9_seed42.pth --direction forward --tta-mirror
    tta_run.sh ft42_mirrev ft9/ckpt/ft9_seed42.pth --direction reverse --tta-mirror
    tta_run.sh b43_mir <base seed43> --direction forward --tta-mirror
    tta_run.sh b43_mirrev <base seed43> --direction reverse --tta-mirror
    python tile_score_control.py
"""
import os
import numpy as np, tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
TILE = 2136


def load(case, stems):
    return np.mean([tifffile.imread(os.path.join(HERE, "tta", case, f"mac9_{s}.tif")).astype(np.float32) for s in stems], 0)


for case in ("pherc0139-w016", "pherc0814-46527"):
    F, R = load(case, ("ft42_mir", "b43_mir")), load(case, ("ft42_mirrev", "b43_mirrev"))
    H, W = F.shape
    best = None
    for y in range(0, max(1, H - TILE // 2), TILE // 2):
        for x in range(0, max(1, W - TILE // 2), TILE // 2):
            f, r = F[y:y + TILE, x:x + TILE], R[y:y + TILE, x:x + TILE]
            m = (f > 0) & (r > 0)
            if m.mean() < 0.5:
                continue
            af, ar = ((f > 127) & m).sum() / m.sum(), ((r > 127) & m).sum() / m.sum()
            if best is None or abs(af - ar) > best[0]:
                best = (abs(af - ar), af, ar, y, x)
    print(f"{case}: best tile (y {best[3]}, x {best[4]}) fwd {best[1]:.3f} rev {best[2]:.3f} score {best[0]:.3f}")
