import numpy as np
import pytest
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from robustness.conditions import Condition, ImageDegrader
from robustness.degradations import (SpotConfig, bright_spots, placement_regions, select_spot_centres,
                                     spot_alpha, spot_candidates)
from robustness.quality import psnr, ssim


def smooth_image(seed=0, size=96):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    base = np.stack([0.5 + 0.3 * np.sin(6 * xx + c) * np.cos(4 * yy) for c in range(3)])
    return torch.from_numpy(np.clip(base + 0.05 * rng.standard_normal(base.shape), 0, 1)).float()


def disk(size=128, radius=30, centre=(64, 64)):
    yy, xx = np.mgrid[0:size, 0:size]
    return (yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= radius ** 2


def test_ssim_matches_skimage():
    x = smooth_image(0)
    y = (x + 0.1 * torch.randn_like(x)).clamp(0, 1)
    ours = float(ssim(x, y))
    ref = structural_similarity(x.permute(1, 2, 0).numpy().astype(np.float64), y.permute(1, 2, 0).numpy().astype(np.float64),
                                channel_axis=-1, data_range=1.0, gaussian_weights=True, sigma=1.5,
                                use_sample_covariance=False)
    assert ours == pytest.approx(ref, abs=1e-6)
    assert float(ssim(x, x)) == pytest.approx(1.0, abs=1e-9)


def test_ssim_and_psnr_batched_match_single():
    x = torch.stack([smooth_image(1), smooth_image(2)])
    y = (x + 0.05 * torch.randn_like(x)).clamp(0, 1)
    batched = ssim(x, y)
    assert batched.shape == (2,)
    assert float(batched[1]) == pytest.approx(float(ssim(x[1], y[1])), abs=1e-12)
    ref = peak_signal_noise_ratio(x[0].numpy().astype(np.float64), y[0].numpy().astype(np.float64), data_range=1.0)
    assert float(psnr(x, y)[0]) == pytest.approx(ref, abs=1e-6)
    assert float(psnr(x[0], x[0])) == float('inf')


def test_regions_partition_image_and_band_width():
    lesion = disk()
    regions = placement_regions(lesion, band_px=5)
    parts = [regions[k] for k in ('boundary', 'interior', 'background')]
    assert np.array_equal(parts[0] | parts[1] | parts[2], np.ones_like(lesion))
    assert not (parts[0] & parts[1]).any() and not (parts[0] & parts[2]).any() and not (parts[1] & parts[2]).any()
    assert regions['interior'][64, 64] and not regions['boundary'][64, 64]
    assert regions['boundary'][64, 64 + 30] and regions['boundary'][64, 64 + 34]   # on / just outside the edge
    assert regions['background'][64, 64 + 40] and regions['interior'][64, 64 + 20]
    assert regions['interior'].sum() < lesion.sum()


def test_regions_without_lesion_edge():
    regions = placement_regions(np.zeros((16, 16), bool), band_px=3)
    assert not regions['boundary'].any() and not regions['interior'].any() and regions['background'].all()


def test_select_centres_respects_click_margin_and_separation():
    cfg = SpotConfig(count=5, click_margin=3.0, min_separation=2.0)
    region = np.ones((200, 200), bool)
    cands = spot_candidates(region, np.random.default_rng(0), cfg.max_candidates)
    click = (100.0, 100.0)
    centres = select_spot_centres(cands, click, radius=6, cfg=cfg)
    assert len(centres) == 5
    for r, c in centres:
        assert (r - 100) ** 2 + (c - 100) ** 2 >= (3 * 6) ** 2
    for i, a in enumerate(centres):
        for b in centres[i + 1:]:
            assert (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 >= (2 * 6) ** 2


def test_select_centres_relaxes_separation_in_small_region():
    cfg = SpotConfig(count=5)
    region = np.zeros((100, 100), bool)
    region[10:13, 10:13] = True   # 9 pixels: no room for 5 well-separated spots
    cands = spot_candidates(region, np.random.default_rng(1), cfg.max_candidates)
    centres = select_spot_centres(cands, (90.0, 90.0), radius=4, cfg=cfg)
    assert len(centres) == 5 and len(set(centres)) == 5
    assert all(region[r, c] for r, c in centres)


def test_spot_alpha_and_blend():
    alpha = spot_alpha([(10, 20), (10, 20)], radius=3, height=40, width=50)
    assert float(alpha.max()) == pytest.approx(1.0)
    assert float(alpha[10, 20]) == 1.0
    assert float(alpha[39, 0]) < 1e-6
    x = smooth_image(3, size=40)[:, :, :40]
    alpha = spot_alpha([(20, 20)], radius=4, height=40, width=40)
    y = bright_spots(x, alpha)
    assert torch.all(y >= x - 1e-7) and torch.all(y <= 1.0)
    assert torch.allclose(y[:, 20, 20], torch.ones(3))
    assert torch.allclose(y[:, 0, 39], x[:, 0, 39], atol=1e-6)


def test_degrader_is_deterministic_and_monotone():
    image = smooth_image(4, size=128)
    lesion = disk()
    make = lambda: ImageDegrader(image, lesion, click_xy=(64.0, 64.0), split_id=1, index=7, seed=0)
    a, b = make(), make()
    noise = Condition('noise', 0.1)
    assert torch.equal(a.apply(noise)[0], b.apply(noise)[0])
    ssims = [float(ssim(image, a.apply(Condition('noise', s))[0])) for s in (0.02, 0.05, 0.1, 0.2)]
    assert all(s1 > s2 for s1, s2 in zip(ssims, ssims[1:]))

    spots = [Condition('spots', r, placement='boundary') for r in (2.0, 4.0, 8.0)]
    outs = [a.apply(c) for c in spots]
    assert torch.equal(outs[1][0], b.apply(spots[1])[0])
    assert all(info['n_spots'] == 5 for _, info in outs)
    assert outs[0][1]['spot_frac_boundary'] > 0.9
    fracs = outs[1][1]
    assert fracs['spot_frac_boundary'] + fracs['spot_frac_interior'] + fracs['spot_frac_background'] == pytest.approx(1.0)
    spot_ssims = [float(ssim(image, out)) for out, _ in outs]
    assert all(s1 > s2 for s1, s2 in zip(spot_ssims, spot_ssims[1:]))


def test_condition_names_and_validation():
    assert Condition('clean').name == 'clean'
    assert Condition('noise', 0.05, match='ssim', target=0.9, level=1).name == 'noise_ssim0.9'
    assert Condition('spots', 12.5, placement='interior', match='psnr', target=31.25).name == 'spots-interior_psnr31.25'
    with pytest.raises(ValueError):
        Condition('spots', 3.0)
    with pytest.raises(ValueError):
        Condition('noise', 0.1, placement='random')
