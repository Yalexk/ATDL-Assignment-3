"""Turn the pipeline outputs into the numbers the report asks for, in the report's units and notation.

Reads any of stats.json (robustness.analyze), levels.json (robustness.calibrate) and feature_stats.csv
(robustness.feature_maps); sections whose input is missing are skipped. Writes
  * report_fill.md     - every value the report leaves as ??, ordered by report section;
  * report_tables.tex  - LaTeX bodies for Table 1 (calibration) and Table 2 (H1).
Dice values are in percentage points; p-values are Holm-adjusted.

    python -m robustness.report_numbers --stats results/analysis/stats.json --levels results/levels.json \
        --features results/features/feature_stats.csv --out_dir results/report
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from .analyze import bootstrap_mean_ci

PRIMARY = 'dice05'


def pts(value):
    return f'{100 * value:.1f}'


def ci_pts(mean, lo, hi):
    return f'{pts(mean)} [{pts(lo)}, {pts(hi)}]'


def p_tex(p):
    if p is None or not np.isfinite(p):
        return 'n/a'
    if p >= 1e-3:
        return f'{p:.3f}'
    mantissa, exponent = f'{p:.1e}'.split('e')
    return f'${mantissa}\\times10^{{{int(exponent)}}}$'


def h1_rows(stats, metric, match):
    return sorted(stats['metrics'][metric][f'h1_{match}'], key=lambda r: r['level'])


def section_clean_and_h1(stats, md):
    ssim_rows = h1_rows(stats, PRIMARY, 'ssim')
    psnr_rows = h1_rows(stats, PRIMARY, 'psnr')
    thavg = h1_rows(stats, 'dice_thavg', 'ssim')
    level = stats['placement_level']
    md.append('## §4.1 Clean performance\n')
    if ssim_rows:
        md.append(f"- Clean Dice at threshold 0.5: **{pts(ssim_rows[0]['clean'])}%** "
                  f"(n = {ssim_rows[0]['n_images']} test images)")
    if thavg:
        md.append(f"- Clean threshold-averaged Dice: **{pts(thavg[0]['clean'])}%**")
    md.append(f"- SSIM targets used: {', '.join(format(r['target'], 'g') for r in ssim_rows)}\n")

    md.append('## §4.2 H1 (Dice at threshold 0.5)\n')
    mid = next((r for r in ssim_rows if r['level'] == level), None)
    if mid:
        md.append(f"- At SSIM {mid['target']:g}: spots lower Dice by **{pts(mid['delta_spots'])}** points, "
                  f"noise by **{pts(mid['delta_noise'])}**; "
                  f"d̄ = **{ci_pts(mid['diff_mean'], mid['diff_ci_lo'], mid['diff_ci_hi'])}**, "
                  f"Wilcoxon p (Holm) = {p_tex(mid['wilcoxon_p_holm'])}")
    for r in ssim_rows:
        md.append(f"- SSIM {r['target']:g}: d̄ = {ci_pts(r['diff_mean'], r['diff_ci_lo'], r['diff_ci_hi'])}, "
                  f"CI above zero: {r['diff_ci_lo'] > 0}, supports H1: {r['supports_h1']}")
    for r in psnr_rows:
        md.append(f"- PSNR level {r['level'] + 1} ({r['target']:g} dB): "
                  f"d̄ = {ci_pts(r['diff_mean'], r['diff_ci_lo'], r['diff_ci_hi'])}, supports H1: {r['supports_h1']}")
    supported = (bool(ssim_rows) and all(r['supports_h1'] for r in ssim_rows)
                 and bool(psnr_rows) and all(r['supports_h1'] for r in psnr_rows))
    md.append(f"- **H1 verdict by the report's rule** (CI above zero at every SSIM level and under PSNR matching): "
              f"{'supported' if supported else 'not supported'}\n")
    return ssim_rows, psnr_rows


def section_h2(stats, md):
    entry = stats['metrics'][PRIMARY]
    md.append('## §4.3 H2 (Dice at threshold 0.5)\n')
    for r in entry['h2_per_placement']:
        md.append(f"- {r['placement'].capitalize()}: drop = "
                  f"**{ci_pts(r['delta_mean'], r['delta_ci_lo'], r['delta_ci_hi'])}** points "
                  f"(radius {r['radius']:.1f} px, mean SSIM {r['ssim']:.3f}, n = {r['n_images']})")
    for r in entry['h2_pairwise']:
        md.append(f"- {r['comparison']}: **{ci_pts(r['diff_mean'], r['diff_ci_lo'], r['diff_ci_hi'])}**, "
                  f"Wilcoxon p (Holm) = {p_tex(r['wilcoxon_p_holm'])}")
    f = entry['h2_friedman']
    md.append(f"- Friedman over the three placements: χ² = {f['statistic']:.1f}, p = {p_tex(f['p'])}, "
              f"n = {f['n_images']}")
    md.append(f"- **H2 ordering supported:** {entry['h2_supported']}\n")


def section_mechanism(features_csv, md):
    df = pd.read_csv(features_csv)
    layers = list(dict.fromkeys(df['layer']))
    first, last = layers[0], layers[-1]
    md.append(f'## §4.4 Mechanism (means over {df["image"].nunique()} images, 95% bootstrap CI)\n')
    md.append('| Measure | Layer | Noise | Spots |\n|---|---|---|---|')
    columns = (('top1_share', 'κ, top-1% share (even = 0.01)'), ('top5_share', 'κ₅, top-5% share (even = 0.05)'),
               ('outside_footprint_share', 'φ, share outside footprint'),
               ('spearman_changed', 'Spearman(change, changed pixels)'),
               ('spearman_new_error', 'Spearman(change, new errors)'))
    for column, label in columns:
        if column not in df:
            continue
        for layer in (first, last):
            cells = []
            for kind in ('noise', 'spots'):
                values = df[(df['kind'] == kind) & (df['layer'] == layer)][column].dropna()
                mean, lo, hi = bootstrap_mean_ci(values, 2000, 0)
                cells.append(f'{mean:.3f} [{lo:.3f}, {hi:.3f}]' if values.size else 'n/a')
            md.append(f"| {label} | {'block 0' if layer == first else 'output'} | {cells[0]} | {cells[1]} |")
    if 'footprint_tokens' in df:
        tokens = df[df['layer'] == last]['footprint_tokens'].dropna()
        n_tokens = 64 * 64
        md.append(f"\n- Spot footprint: {tokens.mean():.0f} of {n_tokens} tokens on average, so a spatially even "
                  f"change would put **{1 - tokens.mean() / n_tokens:.2f}** of it outside (compare with φ for noise).")
    md.append('')


def tables_tex(levels, ssim_rows, psnr_rows):
    out = []
    if levels:
        out.append('% Table 1 body (validation set): target & sigma & PSNR & r & PSNR; n/r = target not reached')
        spots = {lvl['target']: lvl for lvl in levels['ssim'].get('spots-random', [])}
        for lvl in levels['ssim']['noise']:
            s = spots.get(lvl['target'])
            noise_cells = f"{lvl['param']:.4f} & {lvl['psnr']:.1f}" if lvl['reachable'] else 'n/r & n/r'
            spot_cells = f"{s['param']:.1f} & {s['psnr']:.1f}" if s and s['reachable'] else 'n/r & n/r'
            out.append(f"{lvl['target']:.2f} & {noise_cells} & {spot_cells} \\\\")
    if ssim_rows:
        out.append('\n% Table 2 body (test set, Dice %, threshold 0.5): matching & noise & spots & d [CI] & p (Holm)')
        for r in ssim_rows:
            out.append(f"SSIM {r['target']:.2f} & {pts(r['noise'])} & {pts(r['spots'])} & "
                       f"{ci_pts(r['diff_mean'], r['diff_ci_lo'], r['diff_ci_hi'])} & {p_tex(r['wilcoxon_p_holm'])} \\\\")
        for r in psnr_rows:
            out.append(f"PSNR, level {r['level'] + 1} & {pts(r['noise'])} & {pts(r['spots'])} & "
                       f"{ci_pts(r['diff_mean'], r['diff_ci_lo'], r['diff_ci_hi'])} & {p_tex(r['wilcoxon_p_holm'])} \\\\")
        out.append(f"% Clean Dice for the Table 2 caption: {pts(ssim_rows[0]['clean'])}%")
    return '\n'.join(out) + '\n'


def section_calibration(levels, md):
    md.append('## Table 1 notes\n')
    meta = levels.get('meta', {})
    md.append(f"- SSIM targets: {meta.get('ssim_targets')}; calibrated on {meta.get('n_images')} validation images")
    for key, lvls in levels['ssim'].items():
        for lvl in lvls:
            if not lvl['reachable']:
                md.append(f"- **{key} did not reach SSIM {lvl['target']:g}**")
    md.append('')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stats', help='stats.json from robustness.analyze')
    parser.add_argument('--levels', help='levels.json from robustness.calibrate')
    parser.add_argument('--features', help='feature_stats.csv from robustness.feature_maps')
    parser.add_argument('--out_dir', default='results/report')
    a = parser.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)

    md = ['# Numbers for the report\n', 'Dice in percentage points; p-values Holm-adjusted; CIs are 95% bootstrap.\n']
    ssim_rows, psnr_rows, levels = [], [], None
    if a.levels:
        with open(a.levels) as f:
            levels = json.load(f)
        section_calibration(levels, md)
    if a.stats:
        with open(a.stats) as f:
            stats = json.load(f)
        ssim_rows, psnr_rows = section_clean_and_h1(stats, md)
        section_h2(stats, md)
    if a.features:
        section_mechanism(a.features, md)

    with open(os.path.join(a.out_dir, 'report_fill.md'), 'w') as f:
        f.write('\n'.join(md))
    with open(os.path.join(a.out_dir, 'report_tables.tex'), 'w') as f:
        f.write(tables_tex(levels, ssim_rows, psnr_rows))
    print(f"wrote {os.path.join(a.out_dir, 'report_fill.md')} and report_tables.tex")


if __name__ == '__main__':
    main()
