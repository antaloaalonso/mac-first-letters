# First Letters sweep plan (Kaggle GPU)

Built from `inventory.json` (`tools/probe.py`, 2026-10-25), a live listing of
dl.ash2txt.org `spiral_datasets/` and S3 `representations/`.

**No eligible scroll has official spiral tracks on a 2.4 µm scan.** All published tracks are on the 8.6/9.4 µm scans,
including PHerc0846A's. So every band run fits on the 9 µm scan and renders that scan at level 0, the native
resolution `ink_9um` was trained for. The two 2.4 µm eligible scans, 0846A and 1203, stay with the Mac
GrowPatch/spiral lanes (`spiral/`, root `segs/`).

| Tier | Scrolls | Inputs | Umbilicus |
|---|---|---|---|
| A | **0125, 0211**, (0826) | tracks + crossings + lasagna | published |
| B | 0191, 0257, 0268, 0358, 0800, 0813 | tracks + crossings + lasagna | estimated from lasagna normals (about 500 px error on crushed 0826; see LOG) |
| C | 0175A, 0175B, 0306B, 0483A, 0483B, 0490A, 0490B, 0846A (9 µm) | tracks + crossings only | CT mask centroid (rough) |
| D | 0846B, 1203 (9 µm), 1218, 1545 | no published tracks (1203/1218/1545 have lasagna) | needs `extract_surface_tracks.py` from surface-m7 first; not wired yet |

Within a tier the order is: published umbilicus first, then 9.362 µm/113 keV (the 0826/0125 family) before
8.640 µm/116 keV, then smaller track stores first (cheaper staging).

## Run queue (about 2.8 GPU-h each; weekly budget about 15 GPU-h)

| # | Run | Params (`RUN` cell) | Purpose |
|---|---|---|---|
| 1 | control | defaults in the notebook (`mode="control"`, PHerc1667, `20251217075048`, level 2, tag `w018`) | Proves the Kaggle stack (torch/CUDA, ink ensemble, renderer). Text must show in forward only. About 0.5 h. |
| 2 | PHerc0125 band | `mode="band", scroll="PHerc0125", volume="20250821151825", level=0, z0=None, tag="p0125_auto"` | First real band. Also times the CUDA fit on a T4. |
| 3 | PHerc0211 band | `mode="band", scroll="PHerc0211", volume="20250821151803", level=0, z0=None, tag="p0211_auto"` | Second tier-A scroll. |
| 4 | resume 2 or 3 | same params + `resume_from="/kaggle/input/<prev output>"` | Only if the fit or the windings ran out of time. |
| 5 | PHerc0826 band | `scroll="PHerc0826", volume="20250821151701", level=0, z0=14600, z1=15600, tag="p0826_b14600"` | Clean parallel layers per the 0826 lane. Avoids that lane's 13000–15000 and the turbulent 8300–11000. |
| 6+ | tier B: 0191 → 0257 → 0358 → 0813 → 0268 → 0800 | `z0=None`, `tag="p<id>_auto"` | |
| later | tier C | `z0=None` | Tracks-only fits are weaker. Expect more sheet jumps. |

`z0=None` picks the 1000-slice window with the most track points lying wholly inside it. On 0826 that rule picks
8500–9500, which the 0826 lane found folded. So once the first auto band of a scroll comes back, check its montage
before spending more bands, and pick later bands by eye from the L3 slices.

## Review rule (from FINDINGS)

A candidate counts only if all of these hold:
- the strokes are in **one** depth direction only (both directions = crack or void rims);
- they survive the void mask;
- they form line structure;
- they are not aligned with fibre bundles or folds.

Threshold for reporting "LIKELY WINNER": ≥10 legible letters inside 4 cm². At 9.36 µm that is about 2140×2140 px of
one winding's TIF.
