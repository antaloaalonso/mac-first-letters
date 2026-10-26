# Why a quiet band at 9.36 µm is weak evidence

The band sweep ranks 2136 px tiles (4 cm² at 9.36 µm) by the asymmetry of predicted ink area between the two papyrus faces:

```text
score = | area(forward > 127) - area(reverse > 127) |        (after void masking)
```

The tiles at the top of that list are the ones a person reviews.

## Test

We applied the same score to the two `RESULTS9.md` held-out regions. Both contain labelled text, and no model was trained on them:
- PHerc0139 w016;
- PHerc0814 46527, a scroll the fine-tune never saw.

The model was the recommended screen, `ft9_seed42` plus base seed43, both with `--tta-mirror`. [`tile_score_control.py`](tile_score_control.py) reproduces the numbers.

| tile | forward area | reverse area | score |
|---|---|---|---|
| held-out text, PHerc0139 w016 (best tile) | 0.097 | 0.018 | **0.079** |
| held-out text, PHerc0814 46527 (best tile) | 0.222 | 0.146 | **0.076** |
| sweep candidates, same models (6 tiles, all reviewed as non-text) | 0.06–0.62 | 0.06–0.39 | **0.17–0.30** |

The candidate tiles are the top-ranked tiles of our own band sweeps on other scrolls. A full-resolution visual review found no letterforms in any of them. They show diffuse or round, millimetre-sized ink blobs, bright-surface patches, or seams along void and mesh edges.

## What this means

- **The tile score does not rank text first.** Real 9 µm text is a few low-amplitude strokes in rows, so its tiles have little ink area in either direction. Blob and surface artifacts light up large areas, and often more in one face than the other.
- **Every artifact tile outscored real text.** That matches what we saw: every top candidate so far has been an artifact.
- **A band with no high-scoring tile is not shown to be empty.** Text could sit below the artifacts in the ranking. Record such bands as "screened", not "negative", and treat the top-ranked tiles as a review queue, not a detector.

## Attempted fix

We also tried a text-specific ranker: a logistic model over saturation, peak-contrast, component-shape, row-periodicity and stroke-width features.
- It passed leave-one-out on these two regions.
- Its top picks on 412 local band windings were all blobs or tile-edge artifacts on visual review.

With only two public held-out text regions at 9 µm, a better ranker cannot be calibrated or validated. More labelled 9 µm text, for example the reserved validation masks of `ink_9um`, is the prerequisite.
