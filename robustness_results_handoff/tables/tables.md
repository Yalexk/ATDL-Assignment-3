## H2: paired differences in the drop in Dice (points, threshold 0.5, SSIM 0.90)

| Comparison | Difference [95% CI] | Wilcoxon p (Holm) |
|---|---|---|
| Boundary − interior | 2.1 [1.1, 3.2] | 1.2e-4 |
| Interior − background | 3.5 [2.7, 4.2] | 7.0e-14 |
| Boundary − background | 5.6 [4.7, 6.4] | 2.1e-27 |

## Mechanism: token-level encoder change (means over images)

| Measure | Noise | Spots |
|---|---|---|
| κ: share of change in top 1% of tokens, block 1 → output | 0.013 → 0.018 | 0.13 → 0.04 |
| φ: share of change outside the spot footprint, block 1 → output | 0.88 → 0.88 (flat) | 0.12 → 0.52, in steps at the global-attention blocks |
