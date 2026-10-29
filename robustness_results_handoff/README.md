# H2 and mechanism results: hand-off for the report

Everything here is written for `medsa_robustness_report_preview.pdf`, in its notation and units: Dice in percentage
points, threshold 0.5, spot placements at SSIM 0.90, and encoder blocks counted 1–12.

## Start here

**To edit the report, follow `INTEGRATION_STEPS.md`:** a numbered checklist that gives, for each step, the exact draft
text to find, what to paste instead, and what still has to be added from the pipeline.

1. **Text:** `latex/report_text.tex` has paste-ready LaTeX. Each block names the `[MISSING]` / `??` it replaces.
   It covers the abstract, §3.5, §4.3, §4.4, the discussion, the conclusion, and two figure environments.
2. **Figures:** `figures/*.pdf`, at single-column width (3.4 in). The `.png` copies are for previewing only.
3. **Tables:** `tables/*.tex` (needs `booktabs`), plus `.csv` and `tables.md` copies.
4. **Anything still `??`:** run `code/report_numbers.py` on the pipeline outputs (see the last section).

## Where each item goes

| Report location | What to use | Status |
|---|---|---|
| Abstract, H2 sentence | `report_text.tex` → ABSTRACT | Ready |
| §3.5, mechanism paragraph | `report_text.tex` → SECTION 3.5 | Ready. **Replaces** the κ (top 5%) / ρ definitions; see "Changes to the method" |
| §3.5, H2 test sentence | `report_text.tex` → SECTION 3.5 | Ready. Adds boundary vs background and Holm |
| §4.3 text | `report_text.tex` → SECTION 4.3 | Pairwise results and verdict ready; the three per-placement drops are `??` |
| Fig. 3 (drop by placement) | `figures/fig3_h2_pairwise.pdf` *or* pipeline `fig2_placement_dice05.pdf` | Ready (see "Choosing Fig. 3") |
| §4.4 text | `report_text.tex` → SECTION 4.4 | Ready, except two Spearman values (`??`) |
| Mechanism figure | `figures/fig_mechanism_layer_spread.pdf` | Ready. New figure, or replaces Fig. 4 (see "The mechanism figure") |
| Optional table | `tables/h2_pairwise.tex`, `tables/mechanism.tex` | Ready. Use only if a table fits better than the figures |
| §5, H2 summary + the Section 3.2 argument | `report_text.tex` → SECTION 5 | Ready |
| §6, H2 and change-map sentences | `report_text.tex` → SECTION 6 | Ready |
| §4.1, §4.2, Tables 1–2, Figs. 2, 4, 5 | Not here: from the pipeline (`stats.json`, `levels.json`, `fig1`–`fig4a`) | Use `report_numbers.py` |

## The results in brief

**H2 is supported.** Paired differences in the drop in Dice at SSIM 0.90, with Holm-corrected Wilcoxon p-values:

| Comparison | Difference [95% CI] (points) | p (Holm) |
|---|---|---|
| Boundary − interior | 2.1 [1.1, 3.2] | 1.2 × 10⁻⁴ |
| Interior − background | 3.5 [2.7, 4.2] | 7.0 × 10⁻¹⁴ |
| Boundary − background | 5.6 [4.7, 6.4] | 2.1 × 10⁻²⁷ |

All three intervals lie above zero, which meets the report's criterion for H2 (§3.5).

**Mechanism: a spot's change starts local, and the global-attention blocks spread it.**

| Measure (block 1 → encoder output) | Noise | Spots |
|---|---|---|
| κ: share of change in the top 1% of tokens (0.01 = even) | 0.013 → 0.018 | 0.13 → 0.04 |
| φ: share of change outside the spot footprint | 0.88 → 0.88 (flat) | 0.12 → 0.52, in steps at blocks 3, 6, 9, 12 |

This answers the open question in §4.4: the change does **not** stay in the footprint, as Gu et al. found for natural
patch corruptions. It spreads through the four global-attention blocks.

### How this fits the argument in §3.2

- **H2 ordering:** this matches the report's own decoder argument (smallest logit margins at the boundary, plus the
  polarity point: bright spots push features towards skin). Interior > background is exactly what the polarity point
  predicts. Keep that argument as the explanation for H2. (The previous version of this hand-off linked H2 to encoder
  spreading instead; that version has been dropped.)
- **Encoder:** "a spot changes a few tokens strongly" is confirmed. "Residual connections keep it in the affected
  tokens" and "attention may ignore it (Gu et al.)" are **not** supported. "A bright token could pass its change on in
  the global blocks" is what happens. The SECTION 5 paragraph in `report_text.tex` says this.

## Changes to the method that the report needs

1. **κ uses the top 1%, not 5%, and is measured at every block, not only the output.** The figures and numbers above use
   the top 1%. The pipeline also computes the top-5% share, so to keep the report's original κ, take it from
   `feature_stats.csv` (`report_numbers.py` prints both) and keep the top-1% figure as supporting evidence.
2. **ρ is not computed anywhere in the code.** The replacement §3.5 text uses φ (share outside the footprint) and the
   Spearman correlation between the change map and changed pixels, both of which the pipeline does compute. The
   alternative is to implement ρ and rerun `robustness.feature_maps`.
3. **H2 has three comparisons with Holm's correction**, not two uncorrected ones. The replacement sentence is in
   `report_text.tex`.

## Choosing Fig. 3

The report's Fig. 3 is meant to show the drop in Dice *per placement*. That chart is the pipeline's
`fig2_placement_dice05.pdf`, which is not in this folder. Two options:

- **A (preferred if available):** use the pipeline's per-placement chart as Fig. 3, and quote the pairwise numbers in
  the text (already written).
- **B (available now):** use `fig3_h2_pairwise.pdf` with the caption in `report_text.tex`. The figure then shows the
  tests directly, and the per-placement drops go in the text.

## The mechanism figure

`fig_mechanism_layer_spread.pdf` is single-column. The report has no slot for it yet: its Fig. 4 is the change *maps*
(the pipeline's `fig4a`). Since the draft is already past four pages, either
- replace Fig. 4 with this figure, and describe the maps in a sentence; or
- keep Fig. 4 and give this figure the next number; or
- drop the figure and use `tables/mechanism.tex`, which takes about a third of the space.

It shows only the first block and the encoder output: the lines join the two points and don't trace the blocks in
between (the caption says so). The full per-block curve comes from `python -m robustness.analyze` with
`--features results/features` (`fig4b_layer_profile.pdf`); prefer that version if the data is available.

## Filling the remaining ?? values

`code/report_numbers.py` turns the pipeline outputs into report-ready values. Run it from the repository root, after
copying it into `robustness/` (it is also already in the repo):

```bash
python -m robustness.report_numbers --stats results/analysis/stats.json --levels results/levels.json \
    --features results/features/feature_stats.csv --out_dir results/report
```

- `report_fill.md` lists every value in report order: clean Dice, the H1 numbers and verdict, the per-placement H2
  drops, Friedman, κ (1% and 5%), φ, Spearman, and the footprint check behind "φ ≈ 0.88 ≈ share outside the footprint".
- `report_tables.tex` has the bodies of Tables 1 and 2, ready to paste between `\midrule` and `\bottomrule`.

Any of the three inputs can be left out, and the script skips the matching sections. It has been tested on synthetic
inputs with the pipeline's file structure, but not yet on the real outputs.

`code/summary_figures.py` redraws the two figures (`python -m robustness.summary_figures --out_dir <dir>`).

## Caveats

- **Footprint interpretation.** φ ≈ 0.88 for noise is consistent with noise changing all tokens evenly, if the spot
  footprint covers about 12% of tokens. `report_numbers.py` prints the actual footprint size so this can be checked
  before it is stated as fact; the §4.4 text currently says "about the share of tokens outside the footprint".
- **Block numbering.** The code numbers blocks 0–11 (global attention at indices 2, 5, 8, 11). Everything here counts
  1–12 to match the report's "twelve blocks", so the global blocks are 3, 6, 9 and 12.
- **One training run.** The intervals reflect variation across test images, not across training runs. The report's
  Limitations paragraph already says this.
- **LaTeX not compiled.** The snippets have not been compiled. `\cite{gu2022}` is a placeholder for reference [11].
