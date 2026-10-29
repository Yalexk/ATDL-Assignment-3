"""Command-line options shared by the robustness scripts."""
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .conditions import Condition
from .data import ISICRobustDataset
from .degradations import SpotConfig
from .repo import build_model, model_args


def add_model_args(parser):
    g = parser.add_argument_group('model')
    g.add_argument('--checkpoint', default=None,
                   help='Med-SA training checkpoint (.pth); omit for zero-shot SAM (requires --mod sam)')
    g.add_argument('--sam_ckpt', required=True, help='SAM base weights, e.g. checkpoint/sam/sam_vit_b_01ec64.pth')
    g.add_argument('--mod', default='sam_adpt', help="'sam_adpt' for Med-SA, 'sam' for zero-shot SAM")
    g.add_argument('--encoder', default='vit_b')
    g.add_argument('--image_size', type=int, default=1024)
    g.add_argument('--out_size', type=int, default=256)
    g.add_argument('--mid_dim', type=int, default=None)
    g.add_argument('--multimask_output', type=int, default=1)
    g.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')


def add_data_args(parser, default_split='test'):
    g = parser.add_argument_group('data')
    g.add_argument('--data_path', required=True, help='ISIC folder containing ISBI2016_ISIC_Part1_*_GroundTruth.csv')
    g.add_argument('--split', choices=('test', 'val'), default=default_split)
    g.add_argument('--prompt_seed', type=int, default=0, help='seed of the fixed click per image')
    g.add_argument('--val_split', type=float, default=0.1, help='must match the training run (split=val only)')
    g.add_argument('--split_seed', type=int, default=0, help='must match the training run (split=val only)')
    g.add_argument('--batch_size', type=int, default=4)
    g.add_argument('--num_workers', type=int, default=4)
    g.add_argument('--limit', type=int, default=None, help='only use the first N images (smoke tests)')


def add_spot_args(parser):
    g = parser.add_argument_group('degradations')
    defaults = SpotConfig()
    g.add_argument('--degrade_seed', type=int, default=0, help='seed of the per-image noise fields and spot positions')
    g.add_argument('--spot_count', type=int, default=defaults.count)
    g.add_argument('--band_px', type=float, default=defaults.band_px)
    g.add_argument('--click_margin', type=float, default=defaults.click_margin)
    g.add_argument('--min_separation', type=float, default=defaults.min_separation)


def spot_config(a):
    return SpotConfig(count=a.spot_count, band_px=a.band_px, click_margin=a.click_margin,
                      min_separation=a.min_separation)


def make_dataset(a, split=None):
    return ISICRobustDataset(a.data_path, split or a.split, image_size=a.image_size, out_size=a.out_size,
                             prompt_seed=a.prompt_seed, val_split=a.val_split, split_seed=a.split_seed,
                             limit=a.limit)


def make_loader(dataset, a):
    return DataLoader(dataset, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
                      pin_memory=str(a.device).startswith('cuda'))


def load_model(a):
    args = model_args(a.sam_ckpt, mod=a.mod, encoder=a.encoder, image_size=a.image_size, out_size=a.out_size,
                      mid_dim=a.mid_dim, multimask_output=a.multimask_output)
    net, config = build_model(args, a.checkpoint, a.device)
    if config is None and a.checkpoint is not None:
        print('note: checkpoint has no stored config; model options were not cross-checked')
    if config is not None and config.get('prompt_type', 'point1') != 'point1':
        print(f"warning: checkpoint was trained with prompt_type={config['prompt_type']}, evaluated with one click")
    if config is not None and a.split == 'val' and config.get('val_split'):
        if (config['val_split'], config['split_seed']) != (a.val_split, a.split_seed):
            raise ValueError('--val_split/--split_seed differ from the training run '
                             f"({config['val_split']}, {config['split_seed']})")
    return net


def parse_condition(spec):
    """'clean' | 'noise:SIGMA' | 'spots:PLACEMENT:RADIUS'."""
    parts = spec.split(':')
    if parts == ['clean']:
        return Condition('clean')
    if parts[0] == 'noise' and len(parts) == 2:
        return Condition('noise', float(parts[1]))
    if parts[0] == 'spots' and len(parts) == 3:
        return Condition('spots', float(parts[2]), placement=parts[1])
    raise ValueError(f'bad condition {spec!r}; use clean, noise:SIGMA or spots:PLACEMENT:RADIUS')


def write_csv(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'wrote {len(rows)} rows to {path}')


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    print(f'wrote {path}')
