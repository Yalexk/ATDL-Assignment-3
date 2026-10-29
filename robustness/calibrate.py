"""Calibrate degradation strengths to matched image quality on the validation images.

For every degradation (noise; spots at each placement) a log-space bisection finds the parameter
(noise sigma, spot radius in px) whose mean SSIM over the validation images equals each target.
PSNR-matched levels for the robustness check are calibrated the same way; by default their targets
are the mean PSNRs that noise reaches at the SSIM levels, so noise is shared and spots are re-matched.

    python -m robustness.calibrate --data_path ISIC --out results/levels.json
"""
import argparse
import math
from dataclasses import asdict

import numpy as np
import torch
from tqdm import tqdm

from .cli import add_spot_args, make_dataset, spot_config, write_json
from .conditions import Condition, ImageDegrader
from .degradations import PLACEMENTS
from .quality import psnr, ssim

SEARCH_RANGES = {'noise': (1e-4, 1.0), 'spots': (0.5, None)}  # spots upper bound: image_size / 2
TOLERANCE = {'ssim': 1e-4, 'psnr': 0.01}


def condition_key(kind, placement):
    return 'noise' if kind == 'noise' else f'spots-{placement}'


class QualityProbe:
    """Mean SSIM / PSNR over a fixed set of images for a given degradation parameter."""

    def __init__(self, degraders):
        self.degraders = degraders
        self.evaluations = 0

    @torch.no_grad()
    def measure(self, kind, placement, param):
        self.evaluations += 1
        ssims, psnrs, spots = [], [], []
        for d in self.degraders:
            out, info = d.apply(Condition(kind, param, placement=placement))
            ssims.append(float(ssim(d.image, out)))
            psnrs.append(float(psnr(d.image, out)))
            if 'n_spots' in info:
                spots.append(info['n_spots'])
        result = {'ssim': float(np.mean(ssims)), 'psnr': float(np.mean(psnrs))}
        if spots:
            result['mean_n_spots'] = float(np.mean(spots))
            result['min_n_spots'] = int(np.min(spots))
        return result


def bisect(measure, metric, target, lo, hi, max_iter=40):
    """Find param in [lo, hi] with measure(param)[metric] == target, assuming it decreases with param.

    'reachable' means the target lies between the values at lo and hi; 'error' is achieved - target
    (spot positions change discretely with the radius, so a reachable target can still be off by > tol).
    """
    tol = TOLERANCE[metric]
    at_lo, at_hi = measure(lo), measure(hi)
    if at_hi[metric] > target:
        return {'param': hi, **at_hi, 'reachable': False, 'error': at_hi[metric] - target}
    if at_lo[metric] < target:
        return {'param': lo, **at_lo, 'reachable': False, 'error': at_lo[metric] - target}
    best = min(({'param': lo, **at_lo}, {'param': hi, **at_hi}), key=lambda r: abs(r[metric] - target))
    for _ in range(max_iter):
        mid = math.sqrt(lo * hi)
        at_mid = measure(mid)
        if abs(at_mid[metric] - target) < abs(best[metric] - target):
            best = {'param': mid, **at_mid}
        if abs(at_mid[metric] - target) <= tol:
            break
        if at_mid[metric] > target:
            lo = mid
        else:
            hi = mid
    return {**best, 'reachable': True, 'error': best[metric] - target}


def feasible_spot_radius(probe, placement, lo, hi):
    """Shrink hi until spots are not lost to the click margin / region size relative to radius lo.

    Large radii exclude more of the image around the click, so beyond some radius fewer than K spots
    fit and the degradation (and SSIM drop) shrinks again; bisection needs the monotone part only.
    """
    reference = probe.measure('spots', placement, lo)['mean_n_spots']
    feasible = lambda r: probe.measure('spots', placement, r)['mean_n_spots'] >= reference - 1e-9
    if feasible(hi):
        return hi
    good, bad = lo, hi
    for _ in range(12):  # geometric bisection on the feasibility boundary (~0.2% precision from [0.5, 512])
        mid = math.sqrt(good * bad)
        if feasible(mid):
            good = mid
        else:
            bad = mid
    return good


def calibrate_levels(probe, kind, placement, metric, targets, image_size):
    lo, hi = SEARCH_RANGES[kind]
    hi = hi if hi is not None else image_size / 2
    if kind == 'spots':
        hi = feasible_spot_radius(probe, placement, lo, hi)
        print(f'  {condition_key(kind, placement):18s} radius search range [{lo:g}, {hi:.4g}] px')
    measure = lambda p: probe.measure(kind, placement, p)
    levels = []
    for level, target in enumerate(targets):
        found = bisect(measure, metric, target, lo, hi)
        levels.append({'level': level, 'target': target, **found})
        status = '' if found['reachable'] else '  (NOT REACHED)'
        if found['reachable'] and abs(found['error']) > TOLERANCE[metric]:
            status = f"  (closest: off by {found['error']:+.2g})"
        print(f"  {condition_key(kind, placement):18s} {metric}={target:<8g} param={found['param']:.5g} "
              f"ssim={found['ssim']:.4f} psnr={found['psnr']:.2f}{status}")
    return levels


def load_degraders(a, spot_cfg):
    dataset = make_dataset(a, split='val')
    degraders = []
    for i in tqdm(range(len(dataset)), desc='loading validation images'):
        s = dataset[i]
        degraders.append(ImageDegrader((s['image'] / 255).to(a.device), s['lesion'].numpy(), s['click'].tolist(),
                                       dataset.split_id, s['index'], a.degrade_seed, spot_cfg))
    return dataset, degraders


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--out_size', type=int, default=256)
    parser.add_argument('--prompt_seed', type=int, default=0, help='clicks are kept free of spots, so must match eval')
    parser.add_argument('--val_split', type=float, default=0.1, help='must match the training run')
    parser.add_argument('--split_seed', type=int, default=0, help='must match the training run')
    parser.add_argument('--limit', type=int, default=None, help='only use the first N validation images')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    add_spot_args(parser)
    parser.add_argument('--ssim_targets', type=float, nargs='+', default=[0.95, 0.90, 0.80])
    parser.add_argument('--psnr_targets', type=float, nargs='+', default=None,
                        help='default: mean PSNR of noise at each SSIM target')
    parser.add_argument('--placements', nargs='+', choices=PLACEMENTS, default=list(PLACEMENTS),
                        help='spot placements to calibrate to the SSIM targets')
    parser.add_argument('--psnr_placements', nargs='+', choices=PLACEMENTS, default=['random'],
                        help='spot placements to calibrate to the PSNR targets')
    parser.add_argument('--out', required=True, help='levels JSON')
    a = parser.parse_args(argv)

    spot_cfg = spot_config(a)
    dataset, degraders = load_degraders(a, spot_cfg)
    probe = QualityProbe(degraders)
    ssim_targets = sorted(a.ssim_targets, reverse=True)

    print('SSIM matching')
    results = {'ssim': {'noise': calibrate_levels(probe, 'noise', None, 'ssim', ssim_targets, a.image_size)}}
    for placement in a.placements:
        results['ssim'][condition_key('spots', placement)] = calibrate_levels(
            probe, 'spots', placement, 'ssim', ssim_targets, a.image_size)

    if a.psnr_targets is None:
        psnr_targets = [round(level['psnr'], 2) for level in results['ssim']['noise']]
    else:
        psnr_targets = sorted(a.psnr_targets, reverse=True)
    print('PSNR matching')
    results['psnr'] = {'noise': calibrate_levels(probe, 'noise', None, 'psnr', psnr_targets, a.image_size)}
    for placement in a.psnr_placements:
        results['psnr'][condition_key('spots', placement)] = calibrate_levels(
            probe, 'spots', placement, 'psnr', psnr_targets, a.image_size)

    results['meta'] = {
        'split': 'val', 'n_images': len(degraders), 'image_indices': dataset.indices,
        'val_split': a.val_split, 'split_seed': a.split_seed, 'prompt_seed': a.prompt_seed,
        'degrade_seed': a.degrade_seed, 'image_size': a.image_size, 'spot': asdict(spot_cfg),
        'ssim_targets': ssim_targets, 'psnr_targets': psnr_targets,
        'psnr_targets_source': 'noise at SSIM targets' if a.psnr_targets is None else 'user',
        'quality': 'mean over images of per-image SSIM (Gaussian 11x11, sigma 1.5, RGB) / PSNR (dB) at model input size',
        'evaluations': probe.evaluations,
    }
    unreachable = [f'{m}/{k}/{lvl["target"]}' for m in ('ssim', 'psnr') for k, levels in results[m].items()
                   for lvl in levels if not lvl['reachable']]
    inexact = [f'{m}/{k}/{lvl["target"]} ({lvl["error"]:+.2g})' for m in ('ssim', 'psnr')
               for k, levels in results[m].items() for lvl in levels
               if lvl['reachable'] and abs(lvl['error']) > TOLERANCE[m]]
    if inexact:
        print('note: matched only approximately:', ', '.join(inexact))
    if unreachable:
        print('WARNING: targets not reached (see README, "Risks"):', ', '.join(unreachable))
    write_json(results, a.out)


if __name__ == '__main__':
    main()
