# Concentrated vs. diffuse degradation (Med-SA, ISIC 2016)

Does the same loss of image quality (matched SSIM) hurt Med-SA more when it is concentrated in a few
bright spots than when it is spread out as Gaussian noise (H1)? Does it matter where the spots land
relative to the lesion (H2)? All commands run from the repository root.

## 1. Retrain without test-set selection

```bash
python train.py -net sam -mod sam_adpt -exp_name msa_isic_robust \
    -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -out_size 256 -b 4 \
    -dataset isic -data_path ../data/ISIC -epochs 40 -val_split 0.1 -split_seed 0 -prompt_seed 0
```

`-val_split 0.1` holds out 90 of the 900 training images (seeded by `-split_seed`, fixed clicks from
`-prompt_seed`). They are used for validation logging instead of the test set. No checkpoint is
selected: `logs/msa_isic_robust_<time>/Model/final_checkpoint.pth` is the epoch-40 model
(`last_checkpoint.pth` is the latest validated epoch). The split is written to `split.json` and stored in the
checkpoint's `config`. Without `-val_split` the original behaviour is unchanged.

## 2. Calibrate degradation levels (no model needed)

```bash
python -m robustness.calibrate --data_path ../data/ISIC --out results/levels.json
```

On the 90 validation images this finds the noise sigma and the spot radius whose mean SSIM is
0.95 / 0.90 / 0.80. It does this for random spots and for spots on the boundary, interior and background.
It also finds levels matched on PSNR, with targets set to the PSNR that noise reaches at each SSIM level.
A target is reported `NOT REACHED` if it can't be bracketed; the sweep then skips it. If spots can't
reach 0.80, re-run with `--ssim_targets 0.95 0.90 0.85` and report that as a design choice.

## 3. Evaluate on the test set

```bash
CKPT=logs/msa_isic_robust_<time>/Model/final_checkpoint.pth
SAM=checkpoint/sam/sam_vit_b_01ec64.pth
python -m robustness.run_sweep --checkpoint $CKPT --sam_ckpt $SAM --data_path ../data/ISIC \
    --levels results/levels.json --out_dir results/medsa
# optional zero-shot SAM reference (frozen backbone, no adapters)
python -m robustness.run_sweep --mod sam --sam_ckpt $SAM --data_path ../data/ISIC \
    --levels results/levels.json --out_dir results/sam_zeroshot
```

`results/medsa/sweep.csv` has one row per (test image, condition). It covers clean; noise and random spots at
every SSIM level; boundary/interior/background spots at the middle SSIM level (`--placements_all_levels`
for all); and noise and random spots at every PSNR level. Each row records the achieved SSIM/PSNR,
Dice/IoU at 0.5, the repository's threshold-averaged Dice/IoU, agreement with the clean prediction, and,
for spots, the number of spots and where their mass fell. Before the sweep, sanity-check the old checkpoint:
`python -m robustness.eval_robust --checkpoint OLD.pth --sam_ckpt $SAM --data_path ../data/ISIC --out results/old_clean.csv`.
Clean `dice_thavg` should be close to the 91.7 reported before. It won't match exactly, because the
clicks are now fixed.

## 4. Mechanism check and analysis

```bash
python -m robustness.feature_maps --checkpoint $CKPT --sam_ckpt $SAM --data_path ../data/ISIC \
    --levels results/levels.json --n_images 20 --out_dir results/features
python -m robustness.analyze --sweep results/medsa/sweep.csv --features results/features --out_dir results/analysis
```

`analyze` writes `stats.md` / `stats.json` (H1 and H2 tests for Dice@0.5, IoU@0.5 and threshold-averaged
Dice) and Figs. 1–4 as PDF and PNG. Fig. 3/4a examples are picked at random (seeded); with
`--example_rule largest_gap` they are the images where spots hurt most relative to noise, which must
then be stated in the caption.

## Definitions (keep the paper consistent with these)

| Item | Definition |
|---|---|
| Image range | Degradations act on the 1024×1024 model input scaled to [0, 1]; the result is rescaled to [0, 255] for SAM |
| Gaussian noise | x′ = clip(x + σz, 0, 1); z ~ N(0, 1) per pixel and channel, fixed per image |
| Bright spots | x′ = x + (1 − x)·min(1, Σₖ Gₖ), K = 5; Gₖ = exp(−‖p − cₖ‖² / 2r²) peaks at 1; r is the calibrated radius |
| Spot centres | Random, fixed order per image and placement; ≥ 3r from the prompt click; ≥ 2r apart where possible (relaxed for small regions) |
| Placements | boundary: ≤ 10 px from the lesion edge on either side; interior / background: inside / outside the lesion and > 10 px from the edge; random: anywhere |
| Click | One positive click, uniformly sampled from the lesion and seeded per image (`--prompt_seed`); identical across conditions |
| SSIM / PSNR | Per image, RGB, Gaussian 11×11 window (σ = 1.5), K1 = 0.01, K2 = 0.03, data range 1; matched on the mean over the 90 validation images |
| Metrics | Dice/IoU at 0.5 (primary) and averaged over thresholds 0.1–0.9 (the repository's metric) at 256×256 |
| H1 test | dᵢ = ΔDice_spots − ΔDice_noise per image; 10k-resample percentile bootstrap 95% CI; Wilcoxon signed-rank; Holm across levels |
| H2 test | ΔDice per placement; Friedman test; pairwise Wilcoxon with Holm |

## Tests

`python -m pytest robustness/tests` checks SSIM/PSNR against scikit-image, the metrics against the
repository's `eval_seg` formulas, the degradations, the calibration search, the condition building and
the statistics. It needs torch, scipy, scikit-image, pandas and matplotlib, but no GPU or data.
