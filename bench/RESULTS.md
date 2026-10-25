# Held-out ink benchmark: Mac streaming input vs. official surface volume

ink_9um step-075000 on the three published validation regions (pixels inside `<case>_validation_mask`, scored against `<case>_inklabels`). `official` = level 2 of the published 2.4 µm surface volume pooled to 21 planes as villa's `prepare_9um_isotropic_input`; `mac` = the published tifxyz re-rendered by `render.py`, streaming only the CT chunks it touches.

| case | val px | ink % | input | model | AUROC | AP | best F1 | P | R |
|---|---|---|---|---|---|---|---|---|---|
| pherc0139-w016 | 178,146 | 23 | official | seed42 | 0.883 | 0.741 | 0.667 | 0.672 | 0.663 |
| pherc0139-w016 | 178,146 | 23 | official | seed43 | 0.914 | 0.762 | 0.713 | 0.617 | 0.843 |
| pherc0139-w016 | 178,146 | 23 | official | seed42+43 | 0.920 | 0.780 | 0.735 | 0.701 | 0.771 |
| pherc0139-w016 | 178,146 | 23 | mac | seed42 | 0.870 | 0.720 | 0.665 | 0.648 | 0.682 |
| pherc0139-w016 | 178,146 | 23 | mac | seed43 | 0.928 | 0.804 | 0.728 | 0.649 | 0.830 |
| pherc0139-w016 | 178,146 | 23 | mac | seed42+43 | 0.925 | 0.792 | 0.738 | 0.700 | 0.780 |
| pherc0814-46527 | 161,051 | 37 | official | seed42 | 0.886 | 0.842 | 0.770 | 0.766 | 0.775 |
| pherc0814-46527 | 161,051 | 37 | official | seed43 | 0.857 | 0.773 | 0.714 | 0.642 | 0.804 |
| pherc0814-46527 | 161,051 | 37 | official | seed42+43 | 0.880 | 0.839 | 0.739 | 0.675 | 0.815 |
| pherc0814-46527 | 161,051 | 37 | mac | seed42 | 0.867 | 0.818 | 0.764 | 0.785 | 0.745 |
| pherc0814-46527 | 161,051 | 37 | mac | seed43 | 0.862 | 0.809 | 0.730 | 0.712 | 0.748 |
| pherc0814-46527 | 161,051 | 37 | mac | seed42+43 | 0.873 | 0.841 | 0.744 | 0.741 | 0.748 |
| pherc1667-w029 | 382,353 | 23 | official | seed42 | 0.901 | 0.759 | 0.695 | 0.618 | 0.794 |
| pherc1667-w029 | 382,353 | 23 | official | seed43 | 0.898 | 0.771 | 0.722 | 0.690 | 0.758 |
| pherc1667-w029 | 382,353 | 23 | official | seed42+43 | 0.925 | 0.816 | 0.754 | 0.733 | 0.777 |
| pherc1667-w029 | 382,353 | 23 | mac | seed42 | 0.892 | 0.757 | 0.688 | 0.619 | 0.774 |
| pherc1667-w029 | 382,353 | 23 | mac | seed43 | 0.876 | 0.757 | 0.692 | 0.722 | 0.663 |
| pherc1667-w029 | 382,353 | 23 | mac | seed42+43 | 0.907 | 0.795 | 0.708 | 0.656 | 0.769 |

**Mean over cases**

| input | model | AUROC | AP | best F1 |
|---|---|---|---|---|
| official | seed42 | 0.890 | 0.781 | 0.711 |
| official | seed43 | 0.890 | 0.769 | 0.716 |
| official | seed42+43 | 0.908 | 0.812 | 0.743 |
| mac | seed42 | 0.876 | 0.765 | 0.706 |
| mac | seed43 | 0.889 | 0.790 | 0.717 |
| mac | seed42+43 | 0.902 | 0.809 | 0.730 |

**Input and prediction agreement** (inside the validation mask)

| case | input NCC, centre layer | ensemble prediction r, official vs mac |
|---|---|---|
| pherc0139-w016 | 0.912 | 0.951 |
| pherc0814-46527 | 0.941 | 0.976 |
| pherc1667-w029 | 0.884 | 0.934 |
