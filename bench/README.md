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
