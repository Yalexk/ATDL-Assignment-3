"""Statistics and figures for the robustness study.

H1: per SSIM (and PSNR) level, the paired difference d_i = dDice_spots,i - dDice_noise,i
    (dDice = Dice_clean - Dice_degraded, random spot placement): mean, 95% percentile-bootstrap CI,
    two-sided Wilcoxon signed-rank test, Holm correction over levels. H1 is supported at a level when
    the CI excludes 0 on the positive side and the Holm-adjusted p < 0.05.
H2: dDice per spot placement at one SSIM level (bootstrap CIs), Friedman test over boundary / interior /
    background, and pairwise Wilcoxon tests (Holm). H2's ordering is supported when boundary > interior
    and interior > background are both significant.

    python -m robustness.analyze --sweep results/medsa/sweep.csv --features results/features --out_dir results/analysis
"""
import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

METRICS = ('dice05', 'iou05', 'dice_thavg')
METRIC_LABELS = {'dice05': 'Dice (threshold 0.5)', 'iou05': 'IoU (threshold 0.5)',
                 'dice_thavg': 'Dice (threshold-averaged)'}
PLACEMENT_ORDER = ('boundary', 'interior', 'background', 'random')
H2_PAIRS = (('boundary', 'interior'), ('interior', 'background'), ('boundary', 'background'))

# Reference data-viz palette (light surface): categorical slots 1-2, ink, recessive axes, blue sequential ramp.
COLORS = {'noise': '#2a78d6', 'spots': '#eb6834'}
INK, INK_SECONDARY, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ffffff'
BLUE_RAMP = LinearSegmentedColormap.from_list('blue_ramp', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
PRED_COLOR, GT_COLOR = '#1baf7a', '#ffffff'


# ----------------------------------------------------------------------------- statistics

def bootstrap_mean_ci(values, n_boot=10000, seed=0, alpha=0.05):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def wilcoxon_p(differences):
    differences = np.asarray(differences, dtype=float)
    if differences.size == 0 or np.all(differences == 0):
        return float('nan')
    return float(stats.wilcoxon(differences).pvalue)


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (NaNs are left out of the family and returned as NaN)."""
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def with_delta(df, metric):
    """Degraded rows with clean_<metric> and delta = clean - degraded for the same image."""
    clean = df[df['kind'] == 'clean'].set_index('index')[metric]
    out = df[df['kind'] != 'clean'].copy()
    out[f'clean_{metric}'] = out['index'].map(clean)
    out['delta'] = out[f'clean_{metric}'] - out[metric]
    return out


def _select(d, kind, match, placement=None, level=None):
    mask = (d['kind'] == kind) & (d['match'] == match)
    if placement is not None:
        mask &= d['placement'] == placement
    if level is not None:
        mask &= d['level'] == level
    return d[mask]


def h1_table(df, metric, match, n_boot=10000, seed=0):
    d = with_delta(df, metric)
    noise = _select(d, 'noise', match)
    spots = _select(d, 'spots', match, placement='random')
    rows = []
    for level in sorted(set(noise['level']) & set(spots['level'])):
        n = _select(noise, 'noise', match, level=level).set_index('index')
        s = _select(spots, 'spots', match, level=level).set_index('index')
        common = n.index.intersection(s.index)
        diff = (s.loc[common, 'delta'] - n.loc[common, 'delta']).to_numpy()
        mean, lo, hi = bootstrap_mean_ci(diff, n_boot, seed + int(level))
        rows.append({
            'match': match, 'level': int(level), 'target': float(n['target'].iloc[0]), 'n_images': len(common),
            'noise_param': float(n['param'].iloc[0]), 'spots_radius': float(s['param'].iloc[0]),
            'noise_ssim': float(n.loc[common, 'ssim'].mean()), 'spots_ssim': float(s.loc[common, 'ssim'].mean()),
            'noise_psnr': float(n.loc[common, 'psnr'].mean()), 'spots_psnr': float(s.loc[common, 'psnr'].mean()),
            'clean': float(n.loc[common, f'clean_{metric}'].mean()),
            'noise': float(n.loc[common, metric].mean()), 'spots': float(s.loc[common, metric].mean()),
            'delta_noise': float(n.loc[common, 'delta'].mean()), 'delta_spots': float(s.loc[common, 'delta'].mean()),
            'diff_mean': mean, 'diff_ci_lo': lo, 'diff_ci_hi': hi, 'wilcoxon_p': wilcoxon_p(diff),
        })
    table = pd.DataFrame(rows)
    if len(table):
        table['wilcoxon_p_holm'] = holm(table['wilcoxon_p'])
        table['supports_h1'] = (table['diff_ci_lo'] > 0) & (table['wilcoxon_p_holm'] < 0.05)
    return table


def h2_tables(df, metric, level, match='ssim', n_boot=10000, seed=0):
    """Return (per-placement table, pairwise table, Friedman result dict) at one level."""
    d = _select(with_delta(df, metric), 'spots', match, level=level)
    per_placement = []
    for k, placement in enumerate(PLACEMENT_ORDER):
        p = d[d['placement'] == placement]
        if p.empty:
            continue
        mean, lo, hi = bootstrap_mean_ci(p['delta'], n_boot, seed + k)
        per_placement.append({'placement': placement, 'n_images': len(p), 'radius': float(p['param'].iloc[0]),
                              'ssim': float(p['ssim'].mean()), 'psnr': float(p['psnr'].mean()),
                              'delta_mean': mean, 'delta_ci_lo': lo, 'delta_ci_hi': hi,
                              'mean_n_spots': float(p['n_spots'].mean()),
                              'spot_frac_boundary': float(p['spot_frac_boundary'].mean()),
                              'spot_frac_interior': float(p['spot_frac_interior'].mean()),
                              'spot_frac_background': float(p['spot_frac_background'].mean())})
    wide = d.pivot_table(index='index', columns='placement', values='delta')
    pairwise, friedman = [], {'statistic': float('nan'), 'p': float('nan'), 'n_images': 0}
    placements = [p for p in ('boundary', 'interior', 'background') if p in wide.columns]
    if len(placements) == 3:
        complete = wide[placements].dropna()
        if len(complete) >= 2:
            result = stats.friedmanchisquare(*[complete[p] for p in placements])
            friedman = {'statistic': float(result.statistic), 'p': float(result.pvalue), 'n_images': len(complete)}
    for k, (a, b) in enumerate(H2_PAIRS):
        if a in wide.columns and b in wide.columns:
            diff = (wide[a] - wide[b]).dropna().to_numpy()
            mean, lo, hi = bootstrap_mean_ci(diff, n_boot, seed + 10 + k)
            pairwise.append({'comparison': f'{a} - {b}', 'n_images': len(diff), 'diff_mean': mean,
                             'diff_ci_lo': lo, 'diff_ci_hi': hi, 'wilcoxon_p': wilcoxon_p(diff)})
    pairwise = pd.DataFrame(pairwise)
    if len(pairwise):
        pairwise['wilcoxon_p_holm'] = holm(pairwise['wilcoxon_p'])
    return pd.DataFrame(per_placement), pairwise, friedman


def h2_supported(pairwise):
    if pairwise.empty:
        return None
    rows = pairwise.set_index('comparison')
    needed = ('boundary - interior', 'interior - background')
    if not all(c in rows.index for c in needed):
        return None
    return bool(all(rows.loc[c, 'diff_ci_lo'] > 0 and rows.loc[c, 'wilcoxon_p_holm'] < 0.05 for c in needed))


# ----------------------------------------------------------------------------- figures

def set_style():
    plt.rcParams.update({
        'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'text.color': INK, 'axes.labelcolor': INK_SECONDARY, 'axes.edgecolor': AXIS,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK_SECONDARY,
        'ytick.labelcolor': INK_SECONDARY, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.5,
        'axes.axisbelow': True, 'legend.frameon': False, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })


def save(fig, out_dir, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'))
    plt.close(fig)
    print(f'wrote {name}.pdf/.png')


def fig_quality_curve(df, metric, match, out_dir, n_boot=10000, seed=0):
    """Fig. 1: segmentation score vs achieved image quality for noise and random spots."""
    quality = match  # x axis = the matched quality metric
    clean = df[df['kind'] == 'clean'][metric]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.axhline(clean.mean(), color=MUTED, linewidth=1, linestyle=(0, (4, 3)), label='Clean')
    series = (('noise', 'Gaussian noise', None), ('spots', 'Bright spots', 'random'))
    ends = []
    for k, (kind, label, placement) in enumerate(series):
        sub = _select(df, kind, match, placement=placement)
        if sub.empty:
            continue
        points = []
        for level, g in sub.groupby('level'):
            mean, lo, hi = bootstrap_mean_ci(g[metric], n_boot, seed + 100 * k + int(level))
            points.append((g[quality].replace(np.inf, np.nan).mean(), mean, lo, hi))
        points.sort()
        x, y, lo, hi = map(np.array, zip(*points))
        ax.errorbar(x, y, yerr=[y - lo, hi - y], color=COLORS[kind], linewidth=2, elinewidth=1, capsize=0,
                    marker='o', markersize=5, markeredgecolor=SURFACE, markeredgewidth=1, label=label, zorder=3)
        ends.append((x[0], y[0], label, COLORS[kind]))
    # Direct labels at each series' most degraded point: the lower series below its marker, the other above.
    for rank, (x0, y0, label, _) in enumerate(sorted(ends, key=lambda e: e[1])):
        below = rank == 0 and len(ends) > 1
        ax.annotate(label, (x0, y0), xytext=(6, -6 if below else 6), textcoords='offset points',
                    va='top' if below else 'bottom', color=INK_SECONDARY, fontsize=7)
    ax.set_xlabel('Mean SSIM (test set)' if quality == 'ssim' else 'Mean PSNR (dB, test set)')
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.legend(loc='lower right')
    save(fig, out_dir, f'fig1_{metric}_vs_{quality}')


def fig_placement(per_placement, metric, level_label, out_dir):
    """Fig. 2: dDice by spot placement with 95% bootstrap CIs."""
    if per_placement.empty:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    x = np.arange(len(per_placement))
    y = per_placement['delta_mean'].to_numpy()
    err = [y - per_placement['delta_ci_lo'].to_numpy(), per_placement['delta_ci_hi'].to_numpy() - y]
    ax.bar(x, y, width=0.6, color=COLORS['spots'], edgecolor=SURFACE, linewidth=2, zorder=2)
    ax.errorbar(x, y, yerr=err, fmt='none', ecolor=INK_SECONDARY, elinewidth=1, capsize=0, zorder=3)
    for xi, yi, lo, hi in zip(x, y, per_placement['delta_ci_lo'], per_placement['delta_ci_hi']):
        negative = yi < 0  # label beyond the whisker on the side the bar points to
        ax.annotate(f'{yi:.3f}', (xi, lo if negative else hi), xytext=(0, -2 if negative else 2),
                    textcoords='offset points', ha='center', va='top' if negative else 'bottom', fontsize=7,
                    color=INK_SECONDARY)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(x, [p.capitalize() for p in per_placement['placement']])
    ax.tick_params(axis='x', length=0)
    short = METRIC_LABELS[metric].split(' (')[0]
    ax.set_ylabel(f'Δ{short} (clean − spots)')
    ax.set_title(f'Spot placement at {level_label}', loc='left', color=INK)
    save(fig, out_dir, f'fig2_placement_{metric}')


def _dice(pred, label):
    pred, label = pred.astype(bool), label.astype(bool)
    return (2 * (pred & label).sum() + 1e-4) / (pred.sum() + label.sum() + 1e-4)


def load_feature_examples(features_dir):
    with open(os.path.join(features_dir, 'feature_meta.json')) as f:
        meta = json.load(f)
    names = [c['condition'] for c in meta['conditions']]
    noise = next(n for n in names if n.startswith('noise'))
    spots = next(n for n in names if n.startswith('spots'))
    examples = []
    for fname in sorted(os.listdir(features_dir)):
        if fname.endswith('.npz'):
            data = dict(np.load(os.path.join(features_dir, fname)))
            data['name'] = fname[:-4]
            data['gap'] = _dice(data[f'{noise}/pred'], data['label']) - _dice(data[f'{spots}/pred'], data['label'])
            examples.append(data)
    return meta, noise, spots, examples


def choose_examples(examples, n, rule, seed=0):
    if rule == 'largest_gap':
        return sorted(examples, key=lambda e: -e['gap'])[:n]
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(examples), size=min(n, len(examples)), replace=False)
    return [examples[i] for i in sorted(picks)]


def _show(ax, image, pred=None, label=None):
    ax.imshow(np.transpose(image, (1, 2, 0)))
    if label is not None:
        ax.contour(label, levels=[0.5], colors=GT_COLOR, linewidths=1, linestyles='dashed')
    if pred is not None and pred.any():
        ax.contour(pred.astype(float), levels=[0.5], colors=PRED_COLOR, linewidths=1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def fig_examples(examples, noise, spots, out_dir):
    """Fig. 3: clean / noise / spots inputs with predicted (solid) and ground-truth (dashed) outlines."""
    fig, axes = plt.subplots(len(examples), 3, figsize=(3.4, 1.2 * len(examples) + 0.3), squeeze=False)
    for r, e in enumerate(examples):
        panels = ((e['image'], e['clean_pred'], 'Clean'), (e[f'{noise}/image'], e[f'{noise}/pred'], 'Noise'),
                  (e[f'{spots}/image'], e[f'{spots}/pred'], 'Spots'))
        for c, (image, pred, title) in enumerate(panels):
            ax = axes[r, c]
            _show(ax, image, pred, e['label'])
            ax.text(0.03, 0.04, f'Dice {_dice(pred, e["label"]):.2f}', transform=ax.transAxes, fontsize=6,
                    color=INK, bbox=dict(facecolor=SURFACE, edgecolor='none', alpha=0.8, pad=1))
            if r == 0:
                ax.set_title(title)
    handles = [plt.Line2D([], [], color=PRED_COLOR, linewidth=1.2, label='Prediction'),
               plt.Line2D([], [], color=INK_SECONDARY, linewidth=1, linestyle='dashed', label='Ground truth')]
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    save(fig, out_dir, 'fig3_examples')


def fig_feature_maps(examples, noise, spots, out_dir, layer='neck'):
    """Fig. 4a: per-token feature change at one layer for noise vs spots, shared colour scale."""
    layer_index = list(examples[0]['layers']).index(layer)
    maps = [e[f'{k}/change'][layer_index].astype(float) for e in examples for k in (noise, spots)]
    vmax = float(np.quantile(np.concatenate([m.ravel() for m in maps]), 0.99))
    fig, axes = plt.subplots(len(examples), 4, figsize=(6.8, 1.6 * len(examples) + 0.3), squeeze=False)
    for r, e in enumerate(examples):
        size = e['image'].shape[-1]
        for c, (key, title) in enumerate(((noise, 'Noise'), (spots, 'Spots'))):
            _show(axes[r, 2 * c], e[f'{key}/image'])
            im = axes[r, 2 * c + 1].imshow(e[f'{key}/change'][layer_index].astype(float), cmap=BLUE_RAMP, vmin=0,
                                           vmax=vmax, interpolation='nearest', extent=(0, size, size, 0))
            axes[r, 2 * c + 1].set_xticks([])
            axes[r, 2 * c + 1].set_yticks([])
            for spine in axes[r, 2 * c + 1].spines.values():
                spine.set_visible(False)
            if r == 0:
                axes[r, 2 * c].set_title(title)
                axes[r, 2 * c + 1].set_title(f'Token change ({layer})')
    fig.colorbar(im, ax=axes, shrink=0.8, label='‖Δf‖ / mean ‖f‖')
    save(fig, out_dir, f'fig4a_feature_change_{layer}')


def fig_layer_profile(features_dir, out_dir, n_boot=2000, seed=0):
    """Fig. 4b: concentration (top-1% share) and spread outside the spot footprint, per encoder layer."""
    stats_df = pd.read_csv(os.path.join(features_dir, 'feature_stats.csv'))
    with open(os.path.join(features_dir, 'feature_meta.json')) as f:
        meta = json.load(f)
    layers = list(dict.fromkeys(stats_df['layer']))
    panels = (('top1_share', 'Share of change in top 1% tokens'),
              ('outside_footprint_share', 'Share of change outside spot footprint'))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.2), sharex=True)
    for ax, (column, title) in zip(axes, panels):
        for block in meta.get('global_attention_blocks', []):
            ax.axvline(block, color=GRID, linewidth=3, zorder=0)
        for k, kind in enumerate(('noise', 'spots')):
            sub = stats_df[stats_df['kind'] == kind]
            if sub.empty or column not in sub or sub[column].isna().all():
                continue
            cis = [bootstrap_mean_ci(sub[sub['layer'] == layer][column].dropna(), n_boot, seed + j)
                   for j, layer in enumerate(layers)]
            mean, lo, hi = map(np.array, zip(*cis))
            x = np.arange(len(layers))
            ax.fill_between(x, lo, hi, color=COLORS[kind], alpha=0.15, linewidth=0)
            ax.plot(x, mean, color=COLORS[kind], linewidth=2, label='Gaussian noise' if kind == 'noise' else 'Bright spots')
        ax.set_title(title, loc='left', color=INK)
        ax.set_xticks(np.arange(len(layers)), [str(int(l[5:])) if l.startswith('block') else 'N' for l in layers])
        ax.set_xlabel('Encoder block (N = neck; shaded: global attention)')
    axes[0].legend(loc='best')
    save(fig, out_dir, 'fig4b_layer_profile')


# ----------------------------------------------------------------------------- report

def markdown_table(df, columns, floatfmt='.4f'):
    if df.empty:
        return '_no data_\n'
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + '---|' * len(columns)]
    for _, row in df[columns].iterrows():
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append('nan' if np.isnan(value) else (f'{value:.2e}' if 0 < abs(value) < 1e-3 else
                                                            format(value, floatfmt)))
            else:
                cells.append(str(value))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sweep', required=True, help='sweep.csv from robustness.run_sweep')
    parser.add_argument('--features', default=None, help='output directory of robustness.feature_maps (Figs. 3-4)')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--primary_metric', choices=METRICS, default='dice05')
    parser.add_argument('--placement_level', type=int, default=None, help='default: middle SSIM level')
    parser.add_argument('--n_boot', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_examples', type=int, default=3)
    parser.add_argument('--example_rule', choices=('random', 'largest_gap'), default='random',
                        help='how Fig. 3/4a examples are picked from the feature-map images (state it in the caption)')
    a = parser.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    set_style()

    df = pd.read_csv(a.sweep)
    ssim_levels = sorted(df.loc[(df['kind'] == 'noise') & (df['match'] == 'ssim'), 'level'].unique())
    placement_level = a.placement_level if a.placement_level is not None else (
        int(ssim_levels[len(ssim_levels) // 2]) if ssim_levels else 0)

    report = {'sweep': os.path.abspath(a.sweep), 'primary_metric': a.primary_metric,
              'placement_level': placement_level, 'n_boot': a.n_boot, 'metrics': {}}
    md = [f'# Robustness analysis\n\nSweep: `{a.sweep}`. Paired over images; 95% percentile bootstrap CIs '
          f'({a.n_boot} resamples); two-sided Wilcoxon signed-rank tests with Holm correction.\n']
    for metric in METRICS:
        entry = {}
        md.append(f'\n## {METRIC_LABELS[metric]}\n')
        for match in ('ssim', 'psnr'):
            table = h1_table(df, metric, match, a.n_boot, a.seed)
            entry[f'h1_{match}'] = table.to_dict(orient='records')
            md.append(f'\n### H1, {match.upper()}-matched: diff = Δspots − Δnoise\n\n')
            md.append(markdown_table(table, ['level', 'target', 'n_images', 'noise_ssim', 'spots_ssim', 'noise_psnr',
                                             'spots_psnr', 'clean', 'noise', 'spots', 'diff_mean', 'diff_ci_lo',
                                             'diff_ci_hi', 'wilcoxon_p', 'wilcoxon_p_holm', 'supports_h1']))
        h1_ssim = pd.DataFrame(entry['h1_ssim'])
        h1_psnr = pd.DataFrame(entry['h1_psnr'])
        entry['h1_supported_ssim_levels'] = h1_ssim.loc[h1_ssim['supports_h1'], 'level'].tolist() if len(h1_ssim) else []
        entry['h1_supported_psnr_levels'] = h1_psnr.loc[h1_psnr['supports_h1'], 'level'].tolist() if len(h1_psnr) else []
        md.append(f"\nH1 supported at SSIM levels {entry['h1_supported_ssim_levels']} and PSNR levels "
                  f"{entry['h1_supported_psnr_levels']} (claim H1 only where both hold).\n")

        per_placement, pairwise, friedman = h2_tables(df, metric, placement_level, 'ssim', a.n_boot, a.seed)
        entry['h2_per_placement'] = per_placement.to_dict(orient='records')
        entry['h2_pairwise'] = pairwise.to_dict(orient='records')
        entry['h2_friedman'] = friedman
        entry['h2_supported'] = h2_supported(pairwise)
        md.append(f'\n### H2, spot placement at SSIM level {placement_level}\n\n')
        md.append(markdown_table(per_placement, ['placement', 'n_images', 'radius', 'ssim', 'delta_mean', 'delta_ci_lo',
                                                 'delta_ci_hi', 'mean_n_spots', 'spot_frac_boundary',
                                                 'spot_frac_interior', 'spot_frac_background']))
        md.append(f"\nFriedman (boundary/interior/background): chi2 = {friedman['statistic']:.3f}, "
                  f"p = {friedman['p']:.3g}, n = {friedman['n_images']}\n\n")
        md.append(markdown_table(pairwise, ['comparison', 'n_images', 'diff_mean', 'diff_ci_lo', 'diff_ci_hi',
                                            'wilcoxon_p', 'wilcoxon_p_holm']))
        md.append(f"\nH2 ordering (boundary > interior > background) supported: {entry['h2_supported']}\n")
        report['metrics'][metric] = entry

        if metric == a.primary_metric:
            for match in ('ssim', 'psnr'):
                fig_quality_curve(df, metric, match, a.out_dir, a.n_boot, a.seed)
            target = h1_ssim.loc[h1_ssim['level'] == placement_level, 'target'] if len(h1_ssim) else []
            fig_placement(per_placement, metric,
                          f'SSIM {float(target.iloc[0]):g}' if len(target) else f'level {placement_level}', a.out_dir)

    if a.features:
        meta, noise, spots, examples = load_feature_examples(a.features)
        if examples:
            chosen = choose_examples(examples, a.n_examples, a.example_rule, a.seed)
            report['figure_examples'] = {'rule': a.example_rule, 'images': [e['name'] for e in chosen]}
            fig_examples(chosen, noise, spots, a.out_dir)
            fig_feature_maps(chosen, noise, spots, a.out_dir)
        fig_layer_profile(a.features, a.out_dir, seed=a.seed)

    with open(os.path.join(a.out_dir, 'stats.json'), 'w') as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    with open(os.path.join(a.out_dir, 'stats.md'), 'w') as f:
        f.write(''.join(md))
    print(f"wrote {os.path.join(a.out_dir, 'stats.json')} and stats.md")


if __name__ == '__main__':
    main()
