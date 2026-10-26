# Held-out ink benchmark

Scores `ink_9um` on the three published validation regions of its label set
(`pherc0139-w016`, `pherc0814-46527`, `pherc1667-w029`), comparing the official
surface volume with this repo's streaming render of the same mesh. Results:
[RESULTS.md](RESULTS.md).

```bash
for c in pherc0139-w016 pherc0814-46527 pherc1667-w029; do
  python bench/fetch_labels.py $c bench/data       # labels + validation box from the HF bucket
  python bench/prepare_inputs.py $c bench/data     # official.zarr and mac.zarr over the box + 256 px
  python bench/align_check.py bench/data $c        # input NCC and best integer shift
  bench/run_infer.sh bench/data $c                 # seed42, seed43, ensemble on both inputs
done
python bench/report.py bench/data bench            # RESULTS.md, results.json
```

About 3.5 GB of CT chunks are cached (`CHUNK_CACHE`, default `cache/`). The 18 inference runs
take about 13 minutes on an M5 Pro.

## Resolution sensitivity (9.36 µm scans)

[RESULTS9.md](RESULTS9.md) scores `ink_9um` on the 9.362 µm scans of the two validation segments that have one. It compares that with 2.4 µm level 2, in full and screen50 modes. The result: no collapse, but mean AUROC falls by 0.05 on official 9 µm input, and by 0.085 in the band-sweep setting (screen50 on streamed renders).

```bash
python bench/res9.py prepare bench/data <case>   # 9.36 um inputs + ECC registration onto the label grid
bench/run_infer9.sh bench/data <case>
python bench/res9.py score bench/data <case>
python bench/res9.py control bench/data <case>   # resampling-only control
```

## Fine-tuning for 9.36 µm input

[FT9_RESULTS.md](FT9_RESULTS.md): a 70-minute MPS fine-tune of ink_9um on the native 9.362 µm labels. It raises held-out AUROC in the band-sweep setting from 0.800 to 0.853 (seed42), 0.863 with `--tta-mirror`, better than the base ensemble at half the cost. The checkpoint is too large for git (273 MB); it is attached to the GitHub release.
