"""Report figures from the reported summary numbers (H2 pairwise tests, encoder-layer endpoints).

Drawn in the style of analyze.py, at the ISBI single-column width (3.4 in), with Dice in percentage points and the
report's notation (kappa: top-1% share, phi: share outside the spot footprint). Use this when only the summary
statistics are at hand; with results/features/feature_stats.csv, `python -m robustness.analyze` draws the full
per-layer profile instead.

    python -m robustness.summary_figures --out_dir results/analysis
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from .analyze import AXIS, COLORS, GRID, INK, INK_SECONDARY, MUTED, SURFACE, save, set_style

# H2 pairwise differences in Delta Dice (percentage points): (label, mean, CI low, CI high, Holm-adjusted p)
H2_PAIRWISE = (
    ('Boundary − interior', 2.1, 1.1, 3.2, 1.2e-4),
    ('Interior − background', 3.5, 2.7, 4.2, 7.0e-14),
    ('Boundary − background', 5.6, 4.7, 6.4, 2.1e-27),
)

# Means over images after the first encoder block and at the encoder output (neck). Blocks are counted from 1 as in
# the report, so the global-attention blocks (code indices 2, 5, 8, 11) are blocks 3, 6, 9 and 12.
LAYER_ENDPOINTS = {
    'kappa': {'noise': (0.013, 0.018), 'spots': (0.13, 0.04)},
    'phi': {'noise': (0.88, 0.88), 'spots': (0.12, 0.52)},
}


def _p_label(p):
    mantissa, exponent = f'{p:.1e}'.split('e')
    return f'$p = {mantissa}\\times10^{{{int(exponent)}}}$'


def fig_h2_pairwise(out_dir):
    """Forest plot: pairwise difference in Delta Dice between placements with 95% CI and Holm-adjusted p."""
    fig, ax = plt.subplots(figsize=(3.4, 1.6))
    y = np.arange(len(H2_PAIRWISE))[::-1]
    for yi, (label, mean, lo, hi, p) in zip(y, H2_PAIRWISE):
        ax.plot([lo, hi], [yi, yi], color=COLORS['spots'], linewidth=2, solid_capstyle='round', zorder=2)
        ax.plot(mean, yi, 'o', color=COLORS['spots'], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                zorder=3)
        ax.annotate(f'{mean:.1f} [{lo:.1f}, {hi:.1f}]', (hi, yi), xytext=(6, 3), textcoords='offset points',
                    va='bottom', fontsize=7, color=INK_SECONDARY)
        ax.annotate(_p_label(p), (hi, yi), xytext=(6, -3), textcoords='offset points', va='top', fontsize=7,
                    color=MUTED)
    ax.axvline(0, color=AXIS, linewidth=1, zorder=1)
    ax.set_yticks(y, [row[0] for row in H2_PAIRWISE])
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID, linewidth=0.5)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Difference in Dice drop (points, 95% CI)')
    save(fig, out_dir, 'fig3_h2_pairwise')


def fig_layer_endpoints(out_dir):
    """Slope chart, single column: kappa and phi at block 0 and at the encoder output, noise vs spots."""
    panels = (('kappa', r'(a) Top-1% share $\kappa$', 0.01, 'even (0.01)', (0, 0.15)),
              ('phi', r'(b) Outside footprint $\varphi$', None, None, (0, 1)))
    labels = {'noise': 'Gaussian noise', 'spots': 'Bright spots'}
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9))
    x = np.array([0, 1])
    for ax, (key, title, ref, ref_label, ylim) in zip(axes, panels):
        if ref is not None:
            ax.axhline(ref, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
            ax.annotate(ref_label, (0.5, ref), xytext=(0, -2), textcoords='offset points', ha='center',
                        va='top', fontsize=6, color=MUTED)
        for kind in ('noise', 'spots'):
            start, end = LAYER_ENDPOINTS[key][kind]
            ax.plot(x, [start, end], color=COLORS[kind], linewidth=2, marker='o', markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=labels[kind], zorder=3)
            # Values beside each end; a flat pair is labelled once, on the right.
            # The noise value in (a) sits just above the reference line, so it is lifted clear of it.
            if start != end:
                lift = 3 if ref is not None and kind == 'noise' else 0
                ax.annotate(f'{start:.3g}', (0, start), xytext=(-5, lift), textcoords='offset points', ha='right',
                            va='bottom' if lift else 'center', fontsize=6, color=INK_SECONDARY)
            ax.annotate(f'{end:.3g}', (1, end), xytext=(5, 0), textcoords='offset points', ha='left',
                        va='center', fontsize=6, color=INK_SECONDARY)
        ax.set_title(title, loc='left', color=INK, fontsize=7)
        ax.set_xticks(x, ['Block 1', 'Output'])
        ax.tick_params(axis='x', length=0, labelsize=6.5)
        ax.tick_params(axis='y', labelsize=6.5)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(*ylim)
    axes[1].annotate('steps at global\nblocks 3, 6, 9, 12', (0.5, 0.68), ha='center', va='center', fontsize=6,
                     color=MUTED)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.07), fontsize=6.5)
    fig.subplots_adjust(wspace=0.45)
    save(fig, out_dir, 'fig_mechanism_layer_spread')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out_dir', default='results/analysis')
    a = parser.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    set_style()
    fig_h2_pairwise(a.out_dir)
    fig_layer_endpoints(a.out_dir)


if __name__ == '__main__':
    main()
