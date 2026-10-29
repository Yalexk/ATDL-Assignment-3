from robustness.run_sweep import build_conditions


def level(i, target, param, reachable=True):
    return {'level': i, 'target': target, 'param': param, 'ssim': target, 'psnr': 30.0, 'reachable': reachable,
            'error': 0.0}


LEVELS = {
    'ssim': {
        'noise': [level(0, 0.95, 0.01), level(1, 0.9, 0.02), level(2, 0.8, 0.04)],
        'spots-random': [level(0, 0.95, 5), level(1, 0.9, 9), level(2, 0.8, 20)],
        'spots-boundary': [level(0, 0.95, 4), level(1, 0.9, 8), level(2, 0.8, 30, reachable=False)],
        'spots-interior': [level(0, 0.95, 4), level(1, 0.9, 8), level(2, 0.8, 12)],
    },
    'psnr': {'noise': [level(0, 35.0, 0.02)], 'spots-random': [level(0, 35.0, 2)]},
}


def test_default_sweep_uses_middle_level_for_placements_and_skips_unreachable():
    conditions, calibration, skipped = build_conditions(LEVELS)
    names = [c.name for c in conditions]
    assert names == ['noise_ssim0.95', 'noise_ssim0.9', 'noise_ssim0.8',
                     'spots-random_ssim0.95', 'spots-random_ssim0.9', 'spots-random_ssim0.8',
                     'spots-boundary_ssim0.9', 'spots-interior_ssim0.9',
                     'noise_psnr35', 'spots-random_psnr35']
    assert skipped == []
    assert calibration['spots-boundary_ssim0.9']['calib_ssim'] == 0.9
    assert conditions[6].param == 8.0 and conditions[6].placement == 'boundary' and conditions[6].level == 1


def test_all_levels_and_unreachable_handling():
    conditions, _, skipped = build_conditions(LEVELS, placements_all_levels=True, include_psnr=False)
    assert skipped == ['spots-boundary_ssim0.8']
    assert 'spots-interior_ssim0.8' in [c.name for c in conditions]
    conditions, _, skipped = build_conditions(LEVELS, placements_all_levels=True, include_unreachable=True)
    assert skipped == [] and 'spots-boundary_ssim0.8' in [c.name for c in conditions]
