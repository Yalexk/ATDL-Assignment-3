"""Run the full study on the test set from a calibrated levels file.

Conditions: clean; noise and random-placement spots at every SSIM level (H1); spots at the
boundary / interior / background placements at the middle SSIM level (H2); noise and random spots
at every PSNR level (robustness check). Unreachable calibration targets are skipped.

    python -m robustness.run_sweep --checkpoint CKPT --sam_ckpt SAM --data_path ISIC \
        --levels results/levels.json --out_dir results/medsa
"""
import argparse
import datetime
import json
import subprocess
from dataclasses import fields

import torch

from .cli import (add_data_args, add_model_args, load_model, make_dataset, make_loader, write_csv, write_json)
from .conditions import Condition
from .degradations import SpotConfig
from .evaluate import evaluate
from .repo import REPO_ROOT


def level_condition(key, metric, level):
    kind = 'noise' if key == 'noise' else 'spots'
    placement = None if key == 'noise' else key.split('-', 1)[1]
    return Condition(kind, float(level['param']), placement=placement, match=metric,
                     target=float(level['target']), level=int(level['level']))


def build_conditions(levels, placement_level=None, placements_all_levels=False, include_psnr=True,
                     include_unreachable=False):
    """Return (conditions, calibration info keyed by condition name, skipped names)."""
    chosen = []
    ssim_levels = levels['ssim']
    n_levels = len(ssim_levels['noise'])
    if placement_level is None:
        placement_level = n_levels // 2
    for key, entries in ssim_levels.items():
        for entry in entries:
            if key in ('noise', 'spots-random') or placements_all_levels or entry['level'] == placement_level:
                chosen.append((key, 'ssim', entry))
    if include_psnr:
        for key, entries in levels.get('psnr', {}).items():
            chosen.extend((key, 'psnr', entry) for entry in entries)

    conditions, calibration, skipped = [], {}, []
    for key, metric, entry in chosen:
        condition = level_condition(key, metric, entry)
        if not entry['reachable'] and not include_unreachable:
            skipped.append(condition.name)
            continue
        conditions.append(condition)
        calibration[condition.name] = {'calib_reachable': entry['reachable'], 'calib_error': entry.get('error'),
                                       'calib_ssim': entry['ssim'], 'calib_psnr': entry['psnr']}
    return conditions, calibration, skipped


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    add_data_args(parser)
    parser.add_argument('--levels', required=True, help='levels JSON from robustness.calibrate')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--placement_level', type=int, default=None,
                        help='SSIM level index for the placement comparison (default: middle level)')
    parser.add_argument('--placements_all_levels', action='store_true')
    parser.add_argument('--no_psnr', action='store_true', help='skip the PSNR-matched conditions')
    parser.add_argument('--include_unreachable', action='store_true',
                        help='also run levels whose calibration target was not reached')
    a = parser.parse_args(argv)

    with open(a.levels) as f:
        levels = json.load(f)
    meta = levels['meta']
    spot_cfg = SpotConfig(**{f.name: meta['spot'][f.name] for f in fields(SpotConfig)})
    if meta['image_size'] != a.image_size:
        raise ValueError(f"levels were calibrated at image_size {meta['image_size']}, not {a.image_size}")
    if meta['prompt_seed'] != a.prompt_seed:
        print(f"note: --prompt_seed {a.prompt_seed} differs from calibration ({meta['prompt_seed']})")

    conditions, calibration, skipped = build_conditions(levels, a.placement_level, a.placements_all_levels,
                                                        not a.no_psnr, a.include_unreachable)
    if skipped:
        print('skipping unreachable calibration levels:', ', '.join(skipped))
    print(f'{len(conditions) + 1} conditions (incl. clean):', ', '.join(c.name for c in conditions))

    net = load_model(a)
    loader = make_loader(make_dataset(a), a)
    started = datetime.datetime.now().isoformat(timespec='seconds')
    rows = evaluate(net, loader, conditions, a.device, a.out_size, a.multimask_output,
                    degrade_seed=meta['degrade_seed'], spot_cfg=spot_cfg)
    for row in rows:
        row.update(calibration.get(row['condition'], {}))
    write_csv(rows, f'{a.out_dir}/sweep.csv')
    write_json({
        'started': started, 'finished': datetime.datetime.now().isoformat(timespec='seconds'),
        'git_commit': git_commit(), 'args': vars(a), 'levels_meta': meta, 'skipped': skipped,
        'conditions': [c.as_row() for c in conditions], 'torch': torch.__version__,
    }, f'{a.out_dir}/sweep_meta.json')


if __name__ == '__main__':
    main()
