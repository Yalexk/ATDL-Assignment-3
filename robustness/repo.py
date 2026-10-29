"""Access to the Med-SA repository code without importing utils.py / function.py.

Those modules call cfg.parse_args() at import time, which would parse this package's command
lines, so the robustness scripts only use cfg.parse_args(argv), models.sam and a standalone
copy of dataset/isic.py (importing it as dataset.isic would run dataset/__init__.py -> utils).
"""
import functools
import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_KEYS = ('mod', 'encoder', 'mid_dim', 'image_size', 'out_size', 'multimask_output')


def ensure_repo_on_path():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


@functools.lru_cache(maxsize=None)
def isic_module():
    spec = importlib.util.spec_from_file_location('medsa_isic', REPO_ROOT / 'dataset' / 'isic.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # lets DataLoader workers pickle ISIC2016 by reference
    spec.loader.exec_module(module)
    return module


def model_args(sam_ckpt, mod='sam_adpt', encoder='vit_b', image_size=1024, out_size=256, mid_dim=None,
               multimask_output=1):
    """Build the repository's argparse namespace for SAM construction."""
    ensure_repo_on_path()
    import cfg
    argv = ['-net', 'sam', '-mod', mod, '-encoder', encoder, '-image_size', str(image_size),
            '-out_size', str(out_size), '-multimask_output', str(multimask_output), '-sam_ckpt', str(sam_ckpt)]
    if mid_dim is not None:
        argv += ['-mid_dim', str(mid_dim)]
    return cfg.parse_args(argv)


def load_checkpoint_file(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location)
    except Exception as err:
        # torch >= 2.6 defaults to weights_only=True; our own checkpoints are trusted.
        if 'weights_only' not in str(err):
            raise
        return torch.load(path, map_location=map_location, weights_only=False)


def _canonical(key, value):
    # sam_model_registry maps 'default' to ViT-B.
    return 'vit_b' if key == 'encoder' and value == 'default' else value


def build_model(args, checkpoint, device):
    """Build SAM from args.sam_ckpt, then load a Med-SA training checkpoint (strictly) if given.

    Returns (net in eval mode on device, stored run config or None).
    """
    ensure_repo_on_path()
    from models.sam import sam_model_registry

    net = sam_model_registry[args.encoder](args, checkpoint=args.sam_ckpt)
    config = None
    if checkpoint is not None:
        state = load_checkpoint_file(checkpoint)
        state_dict = state['state_dict'] if 'state_dict' in state else state
        state_dict = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        net.load_state_dict(state_dict, strict=True)
        config = state.get('config')
        if config is not None:
            mismatched = {k: (config.get(k), getattr(args, k)) for k in MODEL_CONFIG_KEYS
                          if k in config and _canonical(k, config[k]) != _canonical(k, getattr(args, k))}
            if mismatched:
                raise ValueError(f'model options differ from the checkpoint config (checkpoint, given): {mismatched}')
    elif args.mod in ('sam_adpt', 'sam_lora', 'sam_adalora'):
        raise ValueError(f'-mod {args.mod} without a checkpoint would use randomly initialised adapters; '
                         'use mod "sam" for zero-shot SAM')
    return net.to(device).eval(), config
