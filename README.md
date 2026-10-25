# mac-first-letters

Hunt for First Letters on an Apple-silicon laptop. No NVIDIA GPU, no bulk volume downloads.

**seed → GrowPatch surface → stream-render from the open-data bucket → ink inference on MPS**

The documented [Vesuvius Challenge](https://scrollprize.org) workflow assumes a CUDA GPU and local copies of multi-hundred-GB volumes. This repo assumes neither. Everything here was run on an M5 Pro MacBook with 64 GB of RAM.

| Stage | Script | What it does |
|---|---|---|
| Rendering | `render.py` | Renders a tifxyz mesh into a surface-volume zarr. It fetches only the 128³ chunks the mesh touches, over plain HTTPS from `vesuvius-challenge-open-data`, into a local cache. It handles any mesh pyramid frame (`--frame-level`) and any sampling level, plus an optional k×k×k mean-pool (`--pool`). |
| Cache | `prune_cache.py`, `prefetch_band.py` | Trim the chunk cache back to a GB budget, oldest-accessed first. Prefetch only the CT chunks that contain predicted surface in a z-band. |
| Seeding | `seedmap.py`, `grow_seeds.sh` | Score surface-prediction slices for thin, parallel, coherent sheets, then grow 50-generation GrowPatch patches from the best seeds. |
| Triage | `triage.sh` | Level-3 (1/8 bandwidth), 13-layer previews in one contact sheet, to reject bad patches before full-resolution rendering. |
| Ink | `run_segments.sh` | Render, then run `ink_9um` (seed42 + seed43 ensemble) in both depth directions, write previews, prune the cache. |
| Review | `preview.py`, `view.py`, `montage.py` | Side-by-side and per-winding review sheets. `montage.py` masks void and crack rims, which were the main source of false "letters". |
| Spiral fit | `spiral/fit_env.sh` | Environment for `villa/spiral-fitting` on MPS or CPU over a 1000-slice band. |

## Validation

- **Renderer vs. official surface volume.** I re-rendered PHerc0800 segment `20251029010146` from its tifxyz. At the centre layer it matches the published surface volume with normalised cross-correlation **0.968** and no offset (`results/renderer_vs_official_pherc0800.png`). The published volumes use the opposite normal sign, so pass `--flip`.
- **Held-out benchmark.** `bench/` scores `ink_9um` on the three published validation regions of its label set, against the published ink labels:
  - `pherc0139-w016`;
  - `pherc0814-46527`;
  - `pherc1667-w029`.

  It compares two inputs: the official surface volume, and this pipeline's streaming render of the same mesh. Results are in `bench/RESULTS.md`.
- **Sanity check.** PHerc1667 (Scroll 4) segment `w018`, rendered from the 2.4 µm volume at level 2, shows rows of Greek letters in the forward direction and noise in reverse. w018 is part of `ink_9um`'s labelled training set, so this only checks orientation and scale; it is not a held-out test. No image is included here; see the data licence below.
- **A/B on the training recipe.** Sampling level 2 directly was visually equivalent to sampling level 1 and mean-pooling 2×2×2, and it uses 8× less bandwidth.
- **Seeding.** Random GrowPatch seeds mostly grew sheet-hopping "starbursts". The official `vc_render_tifxyz` shows the same thing (`results/starburst_official_vs_mine.jpg`). Seeded patches were clean recto 10/12 times on PHerc1203 and 15/19 times on PHerc0846A, versus about 4/12 with random seeds (`results/pherc1203_growpatch_triage.jpg`).

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
git clone --depth 1 https://github.com/ScrollPrize/villa.git        # run_segments.sh uses villa/vesuvius/src
hf download scrollprize/ink_9um hybrid_3d2d-seed42/step-075000.pth hybrid_3d2d-seed43/step-075000.pth --local-dir checkpoints/ink_9um
gh release download latest -R ScrollPrize/villa -p '*macos-arm64.zip' && unzip -d tools VC3D-*.zip   # vc_grow_seg_from_seed
```

If you install `vesuvius` itself instead of using `PYTHONPATH`, `uv sync --extra models-core` in `villa/vesuvius` avoids building the C++ bindings. That extra exists only once [ScrollPrize/villa PR: models-core extra] is merged.

## Usage

```bash
# 1. Pick seeds on a scroll's published surface prediction, then grow capped patches.
python seedmap.py data/PHerc1203_2um/surface.zarr seeds/PHerc1203.txt
./grow_seeds.sh data/PHerc1203_2um/surface.zarr segs/PHerc1203_2um grow_params_clean.json seeds/PHerc1203.txt 30

# 2. Triage at level 3 (1/8 the bandwidth); look at triage/PHerc1203_2um.jpg and delete bad patches.
./triage.sh segs/PHerc1203_2um \
  https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr 2

# 3. Render at level 2 (9.6 µm, what ink_9um expects) and run the ensemble both ways.
./run_segments.sh segs/PHerc1203_2um \
  https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr \
  PHerc1203_2um --level 2 --frame-level 2
```

GrowPatch quirks:
- `vc_grow_seg_from_seed` rejects the published surface-prediction `metadata.json` because it has no `scan` key. Rename that file and set `voxelsize` in the params JSON.
- It needs every pyramid level present locally. That's about 4.5 GB per 2.4 µm scroll with `normal-grids`.

### Spiral fitting on a Mac

`villa/spiral-fitting` runs on MPS once ScrollPrize/villa PR "spiral-fitting: make weighted track sampling work on MPS" is merged. Without it:
- `torch.multinomial` asks Metal for an 80 GB buffer and aborts;
- the crossing seed comes back negative;
- driver memory grows by 0.35 GB per step, because MPS keeps a compiled graph with its own buffers for every new track-sample size.

With it, a 1000-slice PHerc0846A band (144k tracks) runs at up to about 1.1 it/s on an M5 Pro, vs. 0.47 it/s on CPU. It holds about 4.5 GB between steps and peaks around 10 GB. A ~2 MB/step residual remains, so for fits much longer than ~10k steps, resume from the headless checkpoints (`FIT_SPIRAL_RESUME_PATH`).

```bash
source spiral/fit_env.sh
cd villa/spiral-fitting && FIT_SPIRAL_CONFIG_OVERRIDES="{$BASE}" uv run python fit_spiral.py --dataset ../../spiral/PHerc0846A
```

The fit dataset was built from public representations only:
- umbilicus: `vc_gen_umbilicus` on the published normal grids;
- tracks: `extract_surface_tracks.py --backend cpu` on the level-2 surface prediction, about 1 h per 1000 slices;
- verified patches: the seeded GrowPatch patches.

## Negative results

About 200 cm² of clean GrowPatch surface on PHerc0800, PHerc1203 and PHerc0846A was scanned with the ensemble. None of it showed line-structured text. Every candidate "letter" traced back to one of two things:
- void or crack rims;
- strokes that appear in *both* depth directions, which means torn papyrus.

Hecate 9.6 µm was no better than `ink_9um` on the control and about 10× slower on MPS.

## Data licence

Scroll data is CC BY-NC 4.0. Scrolls 1–4 and Fragments 1–6 carry extra EduceLab-Scrolls terms, including no public revelation of hidden text without approval. For that reason this repo ships no ink images from those scrolls.

## Licence

Code: MIT (see `LICENSE`).
