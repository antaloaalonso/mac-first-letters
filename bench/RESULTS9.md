# Resolution sensitivity: ink_9um on 9.36 µm scans vs. 2.4 µm level 2

`ink_9um` was trained on 2.4 µm scans reduced to ~9.6 µm (level 2, 4× z-mean). Band sweeps on scrolls that only have a 9.36 µm scan run it on level 0 of that scan. This measures how much held-out accuracy that costs, on the two `ink_9um` validation regions that also have a 9.362 µm scan of the same segment. PHerc1667 w029 has none.

Method:
- The published 9.362 µm surface volumes, and our streamed renders of the published 9.362 µm meshes, are inferred at 50% overlap.
- Each prediction is warped onto the label grid with an affine map fitted by ECC between the two scans' surface volumes. The fitted scale is 0.972–0.979, against 9.362/9.596 = 0.976 expected. Cross-scan NCC is 0.81 (pherc0139-w016), 0.65 (pherc0814-46527), and the NCC peak is sharp at zero shift.
- Predictions are scored inside the validation mask, like `RESULTS.md`.
- Control row: the 2.4 µm prediction pushed through the same warp and back, to show that resampling alone costs ≤0.006 AUROC.

| input | mode | pherc0139-w016 AUROC / AP | pherc0814-46527 AUROC / AP | mean AUROC | mean AP |
|---|---|---|---|---|---|
| 2.4 µm L2, official surface volume | full (seed42+43) | 0.920 / 0.780 | 0.880 / 0.839 | **0.900** | 0.810 |
| 2.4 µm L2, official surface volume | screen50 (seed42) | 0.883 / 0.741 | 0.886 / 0.842 | **0.884** | 0.791 |
| 2.4 µm L2, streamed render (render.py) | full (seed42+43) | 0.925 / 0.792 | 0.873 / 0.841 | **0.899** | 0.817 |
| 2.4 µm L2, streamed render (render.py) | screen50 (seed42) | 0.870 / 0.720 | 0.867 / 0.818 | **0.868** | 0.769 |
| 9.36 µm L0, official surface volume (28 layers) | full (seed42+43) | 0.872 / 0.733 | 0.836 / 0.770 | **0.854** | 0.752 |
| 9.36 µm L0, official surface volume (28 layers) | screen50 (seed42) | 0.819 / 0.671 | 0.823 / 0.764 | **0.821** | 0.717 |
| 9.36 µm L0, streamed render, 31 layers (band-sweep input) | full (seed42+43) | 0.864 / 0.709 | 0.810 / 0.754 | **0.837** | 0.731 |
| 9.36 µm L0, streamed render, 31 layers (band-sweep input) | screen50 (seed42) | 0.815 / 0.657 | 0.784 / 0.728 | **0.800** | 0.693 |
| 9.36 µm L0, streamed render, 28 layers | full (seed42+43) | 0.859 / 0.717 | 0.816 / 0.755 | **0.837** | 0.736 |
| 9.36 µm L0, streamed render, 28 layers | screen50 (seed42) | 0.824 / 0.684 | 0.784 / 0.727 | **0.804** | 0.705 |
| control: 2.4 µm official, warped to the 9.36 µm grid and back | full (seed42+43) | 0.921 / 0.782 | 0.880 / 0.841 | **0.900** | 0.812 |
| control: 2.4 µm official, warped to the 9.36 µm grid and back | screen50 (seed42) | 0.883 / 0.744 | 0.886 / 0.844 | **0.885** | 0.794 |

**Findings**
- **No collapse, but a real loss.** On official input, mean AUROC drops from 0.900 at 2.4 µm level 2 to 0.854 at 9.36 µm (-0.046); AP drops by about 0.06. The resampling control rules out registration and interpolation as the cause.
- **The band-sweep setting loses the most.** screen50 on streamed 9.36 µm renders scores 0.800, against 0.884 for the same mode at 2.4 µm (-0.085). With the full ensemble it scores 0.837.
- **The ensemble matters more at 9.36 µm.** It adds +0.037 AUROC on the sweep input, against +0.016 at 2.4 µm. On 9 µm-only scrolls, confirm any screen50 negatives with a full pass before calling a region empty.
- **Layer count is not the gap.** Rendering 28 instead of 31 layers changes AUROC by ≤0.006. The streamed 9.36 µm input still trails the official 9.36 µm surface volume by 0.01–0.025, a gap we haven't explained yet.
- **What a negative means.** Ranking quality on 9 µm-only scrolls is 6–8 AUROC points lower than at 2.4 µm level 2. A region that shows no letters at 9 µm is weaker evidence of no ink than the same result at 2.4 µm. Where a scroll has both scans, prefer the 2.4 µm one.

Reproduce: `python bench/res9.py prepare bench/data <case>`, then `bench/run_infer9.sh bench/data <case>`, then `python bench/res9.py score bench/data <case>` and `python bench/res9.py control bench/data <case>`.
