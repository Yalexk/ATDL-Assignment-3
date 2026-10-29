# Step-by-step: putting these results into the report

A checklist for editing the report's `.tex` source. Every step names the exact text to find in the current draft
(`medsa_robustness_report_preview.pdf`) and the block of `latex/report_text.tex` to paste in its place. Tick the steps off
as you go. Part A uses only what's in this folder. Part B needs the pipeline output files.

---

## Part A: what you can do now

### Step 1. Preamble

Make sure the report's preamble has these packages (the ISBI template usually already loads `graphicx` and `amsmath`):

```latex
\usepackage{graphicx}   % figures
\usepackage{amsmath}    % \lVert, \Big/
\usepackage{booktabs}   % only if you use the tables in tables/
```

### Step 2. Copy the figures

Copy `figures/fig3_h2_pairwise.pdf` and `figures/fig_mechanism_layer_spread.pdf` into the report's figure folder.
The snippets use `figures/<name>.pdf`; if the report keeps its figures somewhere else, change the path in the two
`\includegraphics` lines. Use the PDFs, not the PNGs.

### Step 3. Fix the citation key

Every `\cite{gu2022}` in `report_text.tex` refers to reference [11] (Gu, Tresp and Qin, ECCV 2022). Replace it with the
key the report already uses for that paper (there are two occurrences, in the §4.4 and §5 blocks).

### Step 4. Abstract

- **Find:** `[MISSING: Main result for H2: boundary vs. interior vs. background.]`
- **Replace with:** the `ABSTRACT` block.
- Leave the H1 and take-away placeholders alone; they depend on H1 (Part B).

### Step 5. §3.5 Evaluation: the H2 test sentence

- **Find:** `H2 uses the same paired tests at SSIM 0.90, for boundary versus interior and interior versus background, and fails if placement makes no difference.`
- **Replace with:** the sentence under `Replaces the H2 sentence in Section 3.5` (it adds boundary vs background and
  Holm's correction).

### Step 6. §3.5 Evaluation: the mechanism paragraph

- **Find:** the last paragraph of §3.5, from `To test the encoder side of the argument in Section 3.2,` up to and including
  `If hotspots and errors were unrelated, ρ would be close to 0.05.` This includes equation (9).
- **Replace with:** the `SECTION 3.5, last paragraph` block.
- **Why:** the draft defines κ over the top 5% of tokens and a measure ρ. The pipeline measured κ over the top **1%**,
  measured it at **every block**, and never computed ρ. The new paragraph defines what was actually measured: κ (1%),
  φ (share outside the spot footprint), and a Spearman correlation in place of ρ.
- **Check:** the new paragraph has its own `equation` environment, which takes the old equation's number (9)
  automatically. If the old equation had a `\label`, add the same label to the new one.

### Step 7. §4.3 H2

- **Find:** from `Fig. 3 shows the drop in Dice for each placement at SSIM 0.90.` through
  `[MISSING: Verdict on H2: does the predicted ordering hold?]`
- **Replace with:** the `SECTION 4.3` block.
- That block still has three `??` (the per-placement drops); they're filled in Step 14.

### Step 8. Fig. 3: choose a version

The draft's Fig. 3 placeholder says "drop in Dice by spot placement". Pick one:

- **Option A (better, needs the pipeline):** use the pipeline's `fig2_placement_dice05.pdf` in the existing Fig. 3 slot.
  Keep the draft's caption ("Drop in Dice at SSIM 0.90 for each spot placement, with 95% bootstrap intervals.") and add
  `\label{fig:h2}` so the §4.3 text's `\ref{fig:h2}` resolves.
- **Option B (available now):** replace the whole Fig. 3 placeholder `figure` environment with the first figure
  environment in the `FIGURES` block of `report_text.tex` (`fig3_h2_pairwise.pdf`, caption included).

Either way, the §4.3 text refers to `\ref{fig:h2}`, so the label must exist.

### Step 9. §4.4 Mechanism

- **Find:** from `Fig. 4 shows the token-level change maps.` through
  `or does it spread through the global attention blocks?]`
- **Replace with:** the `SECTION 4.4` block.
- **Keep** the sentence `Fig. 5 shows typical predictions.` and the Fig. 5 description after it; they're unchanged.
- Two `??` remain (the Spearman values); they're filled in Step 14.

### Step 10. Mechanism figure: choose a placement

The `SECTION 4.4` text refers to `\ref{fig:layers}`. Paste the second figure environment from the `FIGURES` block
(`fig_mechanism_layer_spread.pdf`), and pick where it goes:

- **Replace Fig. 4** (the change-maps placeholder) with it, if the maps won't be ready or page space is short. Then
  delete the draft's Fig. 4 placeholder and its caption.
- **Add it as a new figure** next to Fig. 4, if the maps are coming from the pipeline (`fig4a_feature_change_neck.pdf`).
  Later figures renumber automatically.
- **Use a table instead**, if space is very tight: paste `tables/mechanism.tex` and change the §4.4 text's
  "Fig.~\ref{fig:layers}" to "Table~\ref{tab:mechanism}".

The report is already past four pages, so the first or third option is probably safest.

### Step 11. §5 Discussion

- **Find:** `[MISSING: Summarise the outcome of H1 and H2 in two or three sentences, with the key numbers.]`
- **Replace with:** the first `SECTION 5` paragraph (H2 part). Leave room for one H1 sentence before it (Part B).
- **Find:** `[MISSING: Which steps of the argument in Section 3.2 do the results support, and which do they not?]`
- **Replace with:** the second `SECTION 5` paragraph. It covers H2 and the encoder; add an H1 sentence once H1 is known.

### Step 12. §6 Conclusion

- **Find:** `Where an artefact lands [MISSING: H2 outcome in one sentence].`
- **Replace** only the `[MISSING: ...]` part with the line after the `SECTION 6` comment, so it reads
  "Where an artefact lands matters, with boundary spots most harmful, …".
- **Find:** `[MISSING: One sentence on what the change maps show.]`
- **Replace with:** the change-maps sentence in the same block.

---

## Part B: needs the pipeline output files

Ask Alex for these three files from the pipeline run (they are not in this folder):

| File | Made by | Fills |
|---|---|---|
| `results/analysis/stats.json` | `robustness.analyze` | Clean Dice, all of H1, Table 2, the per-placement H2 drops, Friedman |
| `results/levels.json` | `robustness.calibrate` | Table 1, and whether SSIM 0.80 was reached |
| `results/features/feature_stats.csv` | `robustness.feature_maps` | The Spearman values, κ at 5%, and the footprint check |

Also ask for the pipeline's figures in `results/analysis/`:
- `fig1_dice05_vs_ssim.pdf` for Fig. 2;
- `fig2_placement_dice05.pdf` for Fig. 3, option A;
- `fig4a_feature_change_neck.pdf` for Fig. 4;
- `fig3_examples.pdf` for Fig. 5;
- `fig4b_layer_profile.pdf`, the full per-block version of the mechanism figure.

### Step 13. Generate the numbers

From the repository root (the script is already at `robustness/report_numbers.py`; otherwise copy it there from
`code/`):

```bash
python -m robustness.report_numbers --stats results/analysis/stats.json --levels results/levels.json \
    --features results/features/feature_stats.csv --out_dir results/report
```

This writes `results/report/report_fill.md` (all values, in report order) and `results/report/report_tables.tex`.
It needs pandas, numpy and scipy, which the repo's environment has.

### Step 14. Fill the `??` values

Using `report_fill.md`:

| Report location | Take from `report_fill.md` |
|---|---|
| §4.1 clean Dice and threshold-averaged Dice | "§4.1 Clean performance" |
| §4.1 SSIM 0.80 or 0.85 question | "Table 1 notes" (lists any target not reached) |
| §4.2 drops, d̄, CI, p, levels above zero, PSNR result | "§4.2 H1" |
| §4.2 H1 verdict | "H1 verdict by the report's rule" |
| §4.3 three per-placement drops (from Step 7) | "§4.3 H2", the Boundary / Interior / Background lines |
| §4.4 two Spearman values (from Step 9) | "§4.4 Mechanism", row "Spearman(change, changed pixels)", layer "output" |
| Table 2 caption, "Clean Dice: ??%" | Last comment line of `report_tables.tex` |

### Step 15. Tables 1 and 2

Open `report_tables.tex` and paste each body between the draft table's `\midrule` and `\bottomrule`, replacing the
`??` rows. The columns match the draft tables. `n/r` means a calibration target was not reached; if that happens at
0.80, set the third row's label to the target actually used (the draft's `[MISSING: 0.80 or 0.85]`).

### Step 16. Check the footprint sentence

§4.4 says noise's φ ≈ 0.88 is "about the share of tokens that lie outside the footprint". The last line of
`report_fill.md`'s mechanism section gives the actual footprint share. If it's not close to 0.88, delete that clause.

### Step 17. Optional: keep κ at 5%

If the group would rather keep the draft's original κ (top 5% at the encoder output), `report_fill.md` also prints
κ₅. Then change "1\%" to "5\%" and "0.01" to "0.05" in the §3.5 paragraph, and quote κ₅ in §4.4. The figure stays
top-1%; say so in its caption.

---

## Final checks

- [ ] `\ref{fig:h2}` and `\ref{fig:layers}` (or `\ref{tab:mechanism}`) resolve: no "??" in the compiled PDF.
- [ ] No `\cite{gu2022}` left (Step 3).
- [ ] Search the source for `??` and `[MISSING` in §3.5, §4.3, §4.4, §5 and §6: none should remain after Part B.
- [ ] Figure numbers in the text still match after any figure was added or replaced (Steps 8 and 10).
- [ ] Units: H2 values are Dice points (2.1, not 0.021) everywhere.
- [ ] Block numbering: the text says blocks 3, 6, 9, 12 (counted from 1), not 2, 5, 8, 11.
- [ ] Page count is still within the ISBI limit; if not, use the table option in Step 10.
