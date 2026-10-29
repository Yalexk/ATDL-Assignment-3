import numpy as np
import pandas as pd
import pytest

from robustness.analyze import bootstrap_mean_ci, h1_table, h2_supported, h2_tables, holm


def test_holm_matches_hand_computation():
    adjusted = holm([0.01, 0.04, 0.03, np.nan])
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[2] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.06)   # monotone: max(0.04 * 1, 0.06)
    assert np.isnan(adjusted[3])


def test_bootstrap_ci_contains_mean():
    values = np.random.default_rng(0).normal(1.0, 0.5, size=200)
    mean, lo, hi = bootstrap_mean_ci(values, n_boot=2000)
    assert lo < mean < hi and lo > 0.8 and hi < 1.2


def synthetic_sweep(spot_penalty, n=60, seed=0):
    rng = np.random.default_rng(seed)
    clean = rng.uniform(0.85, 0.95, n)
    rows = []
    for i in range(n):
        rows.append({'index': i, 'kind': 'clean', 'placement': None, 'match': None, 'level': np.nan,
                     'target': np.nan, 'param': 0.0, 'ssim': 1.0, 'psnr': np.inf, 'dice05': clean[i]})
        for level, target in enumerate((0.95, 0.9)):
            drop = 0.01 * (level + 1)
            base = {'index': i, 'match': 'ssim', 'level': level, 'target': target, 'ssim': target, 'psnr': 30.0}
            rows.append({**base, 'kind': 'noise', 'placement': None, 'param': 0.02,
                         'dice05': clean[i] - drop + rng.normal(0, 0.003)})
            rows.append({**base, 'kind': 'spots', 'placement': 'random', 'param': 10.0,
                         'dice05': clean[i] - drop - spot_penalty + rng.normal(0, 0.003)})
            for placement, extra in (('boundary', 0.05), ('interior', 0.03), ('background', 0.0)):
                rows.append({**base, 'kind': 'spots', 'placement': placement, 'param': 10.0,
                             'dice05': clean[i] - extra + rng.normal(0, 0.003), 'n_spots': 5,
                             'spot_frac_boundary': 0.0, 'spot_frac_interior': 0.0, 'spot_frac_background': 1.0})
    return pd.DataFrame(rows)


def test_h1_detects_planted_effect_and_not_null():
    table = h1_table(synthetic_sweep(0.02), 'dice05', 'ssim', n_boot=2000)
    assert list(table['level']) == [0, 1]
    assert table['diff_mean'].to_numpy() == pytest.approx([0.02, 0.02], abs=0.003)
    assert table['supports_h1'].all()
    null = h1_table(synthetic_sweep(0.0), 'dice05', 'ssim', n_boot=2000)
    assert not null['supports_h1'].any()


def test_h2_ordering():
    per_placement, pairwise, friedman = h2_tables(synthetic_sweep(0.0), 'dice05', level=1, n_boot=2000)
    assert list(per_placement['placement']) == ['boundary', 'interior', 'background', 'random']
    assert per_placement.set_index('placement').loc['boundary', 'delta_mean'] == pytest.approx(0.05, abs=0.003)
    assert friedman['p'] < 1e-6 and friedman['n_images'] == 60
    assert h2_supported(pairwise) is True
