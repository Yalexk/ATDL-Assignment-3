import math

import pytest

from robustness.calibrate import bisect


def fake_measure(param):
    value = math.exp(-param)  # decreasing quality
    return {'ssim': value, 'psnr': 40 - 10 * param}


def test_bisect_hits_target():
    found = bisect(fake_measure, 'ssim', 0.9, 1e-4, 10.0)
    assert found['reachable']
    assert found['param'] == pytest.approx(-math.log(0.9), rel=1e-3)
    assert abs(found['error']) <= 1e-4
    found = bisect(fake_measure, 'psnr', 31.5, 0.01, 5.0)
    assert found['psnr'] == pytest.approx(31.5, abs=0.01)


def test_bisect_reports_unreachable_targets():
    too_low = bisect(fake_measure, 'ssim', 1e-9, 0.1, 2.0)
    assert not too_low['reachable'] and too_low['param'] == 2.0
    too_high = bisect(fake_measure, 'ssim', 0.99999, 0.1, 2.0)
    assert not too_high['reachable'] and too_high['param'] == 0.1
