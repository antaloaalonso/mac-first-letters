# Fine-tuning ink_9um for 9.36 µm input

**Training data.** The ink_9um `native9` labels: PHerc0139 w035, w039, w040, w041 and w044, annotated directly on the published 9.362 µm surface volumes, about 5.0M supervised pixels. No held-out segment is used. The base checkpoints had already seen these labels.

**Recipe** (`ft9/ft9.py`, fixed in advance):
- 2000 iterations, batch 16, AdamW, lr 3e-5 one-cycle;
- the original 0.5 label smoothing, BCE + Dice, ±2-plane z jitter, rotations and flips;
- about 70 min per seed on an M5 Pro (MPS).

The final iteration is reported; no checkpoint was selected on held-out data.

**Test.** The same held-out validation regions as `RESULTS9.md`. PHerc0139 w016 is the training scroll; PHerc0814 46527 is a different scroll. All runs at 50% overlap.

### 9.36 µm streamed render, 31 layers (band-sweep input)

| model | w016 AUROC / AP | 46527 AUROC / AP | mean AUROC | mean AP |
|---|---|---|---|---|
| base seed42 (screen50) | 0.815 / 0.657 | 0.784 / 0.728 | 0.800 | 0.693 |
| base seed43 | 0.867 / 0.645 | 0.801 / 0.729 | 0.834 | 0.687 |
| base seed42+43 (full) | 0.864 / 0.709 | 0.810 / 0.754 | 0.837 | 0.731 |
| **ft seed42** | 0.868 / 0.731 | 0.838 / 0.801 | 0.853 | 0.766 |
| ft seed43 | 0.889 / 0.730 | 0.752 / 0.687 | 0.820 | 0.708 |
| **ft seed42+43** | 0.898 / 0.762 | 0.819 / 0.775 | 0.858 | 0.768 |

### 9.36 µm official surface volume

| model | w016 AUROC / AP | 46527 AUROC / AP | mean AUROC | mean AP |
|---|---|---|---|---|
| base seed42 (screen50) | 0.819 / 0.671 | 0.823 / 0.764 | 0.821 | 0.717 |
| base seed43 | 0.859 / 0.637 | 0.800 / 0.719 | 0.830 | 0.678 |
| base seed42+43 (full) | 0.872 / 0.733 | 0.836 / 0.770 | 0.854 | 0.752 |
| **ft seed42** | 0.864 / 0.744 | 0.829 / 0.804 | 0.847 | 0.774 |
| ft seed43 | 0.884 / 0.719 | 0.753 / 0.694 | 0.818 | 0.706 |
| **ft seed42+43** | 0.896 / 0.761 | 0.816 / 0.782 | 0.856 | 0.772 |

### 2.4 µm L2 official surface volume

| model | w016 AUROC / AP | 46527 AUROC / AP | mean AUROC | mean AP |
|---|---|---|---|---|
| base seed42 (screen50) | 0.883 / 0.741 | 0.886 / 0.842 | 0.884 | 0.791 |
| base seed43 | 0.914 / 0.762 | 0.857 / 0.773 | 0.886 | 0.767 |
| base seed42+43 (full) | 0.920 / 0.780 | 0.880 / 0.839 | 0.900 | 0.810 |
| **ft seed42** | 0.919 / 0.835 | 0.882 / 0.815 | 0.900 | 0.825 |
| ft seed43 | 0.934 / 0.832 | 0.853 / 0.743 | 0.894 | 0.788 |
| **ft seed42+43** | 0.936 / 0.859 | 0.873 / 0.795 | 0.905 | 0.827 |
## Findings

- **The fine-tuned seed42 is the best screening model at 9.36 µm.** On the band-sweep input it raises mean AUROC from 0.800 to **0.853** (+0.053) and AP from 0.693 to **0.766**. That beats the base two-model ensemble (0.837 / 0.731) at half its cost. It improves *both* held-out regions, including the scroll it never saw: PHerc0814 goes 0.784 → 0.838.
- **Cross-scroll transfer depends on the seed.** The fine-tuned seed43 improves PHerc0139 (0.867 → 0.889) but loses 0.049 on PHerc0814 (0.801 → 0.752). Only one held-out region is from a different scroll, so the cross-scroll benefit is suggestive, not established.
- **The fine-tuned ensemble has the best mean** (0.858 sweep input, 0.856 official 9 µm, 0.905 at 2.4 µm), but on PHerc0814 it scores below the fine-tuned seed42 alone.
- **2.4 µm input is not harmed.** The fine-tuned seed42 matches the base ensemble's mean AUROC (0.900) with higher AP (0.825 vs. 0.810).
- **Free side result:** base seed43 alone beats base seed42 at 9.36 µm on both regions (sweep input 0.834 vs. 0.800).

## Recommendation for 9 µm band sweeps

1. **Screen with the fine-tuned seed42** (`ft9/ckpt/ft9_seed42.pth`, 138 MB) at 50% overlap, in place of base seed42. Inference flags are unchanged: `--overlap 0.5 --blend-mode hann`, single checkpoint.
2. **Confirm candidates with the base seed42+43 ensemble**, the validated recipe, rather than the fine-tuned ensemble, until more cross-scroll held-out data exists.
3. **Next step:** validate on a third scroll, either by mapping more aligned labels from other scrolls onto their 9 µm scans, or with teacher labels from 2.4 µm predictions.

Reproduce: `fetch_native9.py <case> ft9/data` for w035 w039 w040 w041 w044. Then `ft9.py ft9/data ft9/ckpt/ft9_seed42.pth --lr 3e-5 --iters 2000` (seed43: `--base …seed43… --seed 1`). Then `ft9/eval_ft.sh <tag> <ckpt> [ensemble ckpt]`.

## Test-time augmentation (added Oct 26)

These come from a follow-up pass over the same two held-out regions, using the same warp and scorer. Its ft42 reproduction matched 0.853 exactly. The pass reported every variant it tried, about 24 per input, so the gains below are small and selected on only two regions.

| sweep input (mac9) | mean AUROC | mean AP |
|---|---|---|
| ft seed42 | 0.853 | 0.766 |
| **ft seed42, `--tta-mirror`** (4 flips) | **0.863** | **0.792** |
| ft seed42 + base seed43, both `--tta-mirror` | 0.870 | 0.793 |
| full 8-way dihedral TTA | 0.861 | 0.789 |
| depth-window shifts / pooling (±1, ±2) | 0.851–0.866 | 0.770–0.800 |
| `--direction both` (forward + reverse averaged) | 0.776 | 0.617 |

- **Mirror TTA improves both regions** (w016 0.868 → 0.885, PHerc0814 0.838 → 0.842) and costs 4× inference. It is the standard choice, not the argmax, so it is the one adopted.
- **`--direction reverse` scores at chance (0.52)**, so averaging it with the forward pass halves the signal. Use the forward direction only.
- **The base seed43 partner is weakly selected:** the "free side result" above was measured on the same regions.

Updated recommendation: screen with `ft9_seed42.pth --tta-mirror`.
