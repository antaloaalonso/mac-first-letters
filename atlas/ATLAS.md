# Coverage atlas: surfaces scanned for ink, with negative results

Every surface below was rendered and run through `ink_9um` (seed42+43 ensemble unless noted) in both depth directions. None showed line-structured text. Areas are true mesh surface areas. Heights are scan z in µm; radii are distances from the umbilicus (where one was available), 5th–95th percentile.

**How strong these negatives are:** on ~9 µm scans (the 8.64 and 9.362 µm rows) `ink_9um` loses sensitivity, and the sweep's tile ranking puts real held-out 9 µm text below artifacts ([`bench/SCORE_NOTE.md`](../bench/SCORE_NOTE.md)). Read those rows as "screened", not "shown empty". The 2.4 µm rows are stronger negatives (held-out AUROC 0.90 vs. 0.85 at 9.36 µm, `bench/RESULTS9.md`).

| scroll | scan | batch | kind | surfaces | area cm² | z range (mm) | radius (mm) | result |
|---|---|---|---|---|---|---|---|---|
| PHerc0800 | 8.64 µm | PHerc0800 published auto-grown | published segments | 6 | 10 | 93.8–114.4 | — | no ink (ink_9um, single model) |
| PHerc1203 | 9.362 µm | PHerc1203 published raw | published segments | 7 | 27 | 53.4–77.7 | — | no ink (single model) |
| PHerc1203 | 2.403 µm L2 | PHerc1203 random GrowPatch | GrowPatch, random seeds | 6 | 93 | 0.0–55.4 | — | no letters; starburst patches |
| PHerc1203 | 2.403 µm L2 | PHerc1203 seeded GrowPatch | GrowPatch, seedmap seeds | 20 | 67 | 0.0–40.9 | — | no line-structured text (ensemble) |
| PHerc0846A | 2.403 µm L2 | PHerc0846A seeded GrowPatch | GrowPatch, seedmap seeds | 25 | 88 | 0.0–38.9 | 0.5–23.3 | no text; candidates traced to crack edges / folds |
| PHerc0846A | 2.403 µm L2 | PHerc0846A spiral band z600–1600 | spiral windings (spliced) | 45 | 156 | 5.8–15.5 | 0.6–11.8 | no text; void/crack rims masked |
| PHerc0846A | 2.403 µm L2 | PHerc0846A spiral band z1300–1500 | spiral windings (first_letters_mac.sh) | 6 | 10 | 12.5–14.4 | 10.1–18.0 | no text (600-step validation fit) |
| PHerc0826 | 9.362 µm | PHerc0826 spiral band z13000–14000 | spiral windings | 63 | 382 | 121.7–131.1 | 0.3–21.3 | no candidates; sub-letter speckle only |

| scroll | surfaces | area cm² |
|---|---|---|
| PHerc0800 | 6 | 10 |
| PHerc1203 | 33 | 187 |
| PHerc0846A | 76 | 254 |
| PHerc0826 | 63 | 382 |
| **total** | **178** | **832** |

Per-surface rows (area, z, radius) are in `atlas.json`. Coverage maps: `coverage_PHerc0800.png`, `coverage_PHerc1203.png`, `coverage_PHerc0846A.png`, `coverage_PHerc0826.png`.

**Gaps worth taking next** (read off the coverage maps):
- **PHerc0846A:** spiral fits cover the inner windings (radius < 12 mm) for z 6–15.5 mm, plus one outer ring at z 12.5–14.4 mm. The outer windings (radius 12–24 mm) outside that ring have only scattered GrowPatch coverage.
- **PHerc0826:** one band, z 121.7–131.1 mm. The rest of the scroll height is unscanned.
- **PHerc1203:** patches only. No spiral fit yet, because the scroll has no published tracks or umbilicus; `first_letters_mac.sh` builds both from the surface prediction.

Regenerate with `python atlas/atlas.py <root> atlas/sources.json atlas` after adding batches to `sources.json`.
