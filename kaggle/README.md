# Free-tier Kaggle path: whole-band spiral fit → ink, no hardware needed

`vesuvius_band.ipynb` runs one band of one scroll end to end on Kaggle's free **GPU T4 x2**. It takes the
official published inputs:
- spiral tracks;
- crossings;
- lasagna normals;
- umbilicus.

It runs: CUDA spiral fit → stream-render every fitted winding → `ink_9um` in both depth directions →
void-masked review montages. With a Kaggle account, anyone can search a scroll for free.

## Use

1. Upload `vesuvius_band.ipynb` to Kaggle. Set **Accelerator: GPU T4 x2** and **Internet: ON**.
2. Edit only the `RUN` cell, for example:
   ```python
   mode="band", scroll="PHerc0125", volume="20250821151825", level=0, z0=None, tag="p0125_auto"
   ```
   `z0=None` picks a band automatically. `SWEEP.md` lists a queue for the eligible scrolls, and `inventory.json`
   says what each scroll has published.
3. **Save & Run All.** Outputs land in `/kaggle/working/<tag>/`:
   - `montage/` (review sheets), `pred/` (ink TIFs), `renders/`, `state.json`;
   - the fitted meshes;
   - `checkpoint_fitted.ckpt`.
4. If the session runs out of time, start a new version with `resume_from="/kaggle/input/<previous output>"`.
   Staging and the fit are skipped once meshes exist, and the 1 GB checkpoint is not copied again.

`ink_mode="screen"` is the band default: seed42 only, 25% overlap, both directions, about 4.5× cheaper.
`ink_mode="full"` is the validated seed42+43 ensemble at 50% overlap; use it to confirm candidate windings
(`winding_list=[...]`).

## How it fits in a T4 and a 9-hour session

- **No 13 GB track download.** `src/fetch.py` fetches the small `.vctracks` index arrays, then only the byte
  ranges of `coordinates.i32` that hold tracks inside the band, into a sparse file. That's 4.2 GB for a
  1000-slice PHerc0125 band. The fitter's `.dbm` is a sparse stub with the real SQLite header, size and mtime, so
  the fitter's database signature check accepts it unchanged.
- Lasagna normals: only the band's rows. Crossings: downloaded whole.
- `src/render_gpu.py` is `../render.py` with parallel chunk prefetch and GPU trilinear sampling. It matches
  `render.py` to within ±1 grey level (mean |diff| 2.5e-5). Against the official surface volume on PHerc0814 46527
  at level 2, its centre-layer NCC is 0.94.
- `src/umbilicus_est.py` estimates an umbilicus from lasagna normals for scrolls without a published one.

## Measured on Kaggle (2×T4, 2026-10-25)

| Run | Result |
|---|---|
| Setup: villa clone, both venvs, `vc_spiral` C++ build, checkpoints | 16.9 min, 14.6 of it the ink-venv install; a lighter ink venv (reusing Kaggle's torch) was added afterwards |
| PHerc0125 band z 4500–5500: 1,394,991 tracks staged as 4.24 GB of byte ranges | fit 3000 steps at **2.1 it/s → 23.7 min**; windings [10, 130) written as tifxyz |
| Fit quality (same run) | 55.4% of sampled track points satisfied, 16.7% of tracks |
| Stack check: PHerc1667 w018 (2.399 µm mesh, level 2), full ensemble | 44.7 min total; Greek letter rows visible in the forward direction |
| Same check, `screen` mode | letter rows still visible, noisier |

w018 is part of `ink_9um`'s training labels. It checks the whole Kaggle stack end to end, not generalisation; for
held-out numbers see `../bench/RESULTS.md`. No w018 images are included, because PHerc1667 is under the Scroll 4
data terms.

**Not yet shown:** a band run that completes the ink stage for every winding inside one session. The resume path
that does this was fixed after the PHerc0125 runs and tested locally, but it has not yet been rerun on Kaggle.

## Files

- `vesuvius_band.ipynb`: generated from `src/` by `python build_nb.py`. Rebuild it after editing `src/`.
- `src/`: `fetch.py`, `render_gpu.py`, `ink.py`, `montage.py`, `umbilicus_est.py`, and `pipeline.py` (the stage
  runner, with budget, resume and state).
- `SWEEP.md`, `inventory.json`, `tools/probe.py`: which eligible scrolls have which published inputs, and a run
  order.

`villa` is pinned to `322cb089`.
