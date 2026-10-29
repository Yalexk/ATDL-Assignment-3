"""Token feature-change maps ||f(x_deg) - f(x)|| for noise vs spots (mechanism check).

For a seeded sample of test images and one calibrated SSIM level, records for every ViT block output
(64x64 tokens for 1024 px input) and the encoder neck:
  * the per-token L2 change, relative to the mean clean token norm of that layer;
  * concentration: share of the total change carried by the top 1% / 5% of tokens;
  * share of the change outside the spot footprint (tokens whose 16x16 patch has blend weight >= 0.05),
    measured for both noise and spots against the same image's spot footprint;
  * Spearman correlation between the change map and the token-pooled map of changed mask pixels.
Per-image maps are saved as .npz for figures; summary statistics go to feature_stats.csv.

    python -m robustness.feature_maps --checkpoint CKPT --sam_ckpt SAM --data_path ISIC \
        --levels results/levels.json --out_dir results/features
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm

from .cli import add_data_args, add_model_args, load_model, make_dataset, write_csv, write_json
from .conditions import ImageDegrader
from .degradations import SpotConfig, spot_alpha, select_spot_centres
from .inference import predict
from .quality import ssim
from .run_sweep import level_condition

FOOTPRINT_ALPHA = 0.05


def token_change(clean, degraded):
    """(H, W) L2 change per token, relative to the mean clean token norm. Inputs (1, H, W, C)."""
    change = (degraded - clean).norm(dim=-1)[0]
    return change / clean.norm(dim=-1)[0].mean()


def top_share(values, fraction):
    flat = np.sort(values.ravel())[::-1]
    k = max(1, int(round(fraction * flat.size)))
    total = flat.sum()
    return float(flat[:k].sum() / total) if total > 0 else float('nan')


def pool_to_tokens(image_map, tokens, mode='avg'):
    x = image_map[None, None].float()
    k = image_map.shape[-1] // tokens
    return (F.avg_pool2d(x, k) if mode == 'avg' else F.max_pool2d(x, k))[0, 0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(parser)
    add_data_args(parser)
    parser.add_argument('--levels', required=True)
    parser.add_argument('--level', type=int, default=None, help='SSIM level index (default: middle level)')
    parser.add_argument('--keys', nargs='+', default=['noise', 'spots-random'],
                        help="calibrated degradations from the levels file, e.g. noise spots-random spots-boundary")
    parser.add_argument('--n_images', type=int, default=20)
    parser.add_argument('--selection_seed', type=int, default=0)
    parser.add_argument('--out_dir', required=True)
    a = parser.parse_args(argv)

    with open(a.levels) as f:
        levels = json.load(f)
    meta = levels['meta']
    spot_cfg = SpotConfig(**meta['spot'])
    if meta['image_size'] != a.image_size:
        raise ValueError(f"levels were calibrated at image_size {meta['image_size']}, not {a.image_size}")
    level = a.level if a.level is not None else len(levels['ssim']['noise']) // 2
    conditions = []
    for key in a.keys:
        entry = levels['ssim'][key][level]
        if not entry['reachable']:
            raise ValueError(f'{key} level {level} was not reachable in calibration')
        conditions.append(level_condition(key, 'ssim', entry))
    spots_condition = next((c for c in conditions if c.kind == 'spots'), None)

    net = load_model(a)
    dataset = make_dataset(a)
    chosen = sorted(np.random.default_rng(a.selection_seed).choice(len(dataset), size=min(a.n_images, len(dataset)),
                                                                   replace=False).tolist())
    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    for i in tqdm(chosen, desc='images'):
        s = dataset[i]
        image01 = (s['image'] / 255).to(a.device)
        click = s['click'].to(a.device)
        label = s['label'].to(a.device)
        degrader = ImageDegrader(image01, s['lesion'].numpy(), s['click'].tolist(), dataset.split_id, s['index'],
                                 meta['degrade_seed'], spot_cfg)
        logits, neck, blocks = predict(net, s['image'][None].to(a.device), click[None], a.out_size,
                                       a.multimask_output, block_outputs=True)
        clean_layers = [b.float() for b in blocks] + [neck.permute(0, 2, 3, 1).float()]
        layer_names = [f'block{j:02d}' for j in range(len(blocks))] + ['neck']
        tokens = clean_layers[0].shape[1]
        clean_pred = logits[0, 0] > 0

        footprint = None
        alpha_tokens = None
        if spots_condition is not None:
            centres = select_spot_centres(degrader.candidates(spots_condition.placement), degrader.click_xy,
                                          spots_condition.param, spot_cfg)
            alpha = spot_alpha(centres, spots_condition.param, a.image_size, a.image_size, device=a.device)
            alpha_tokens = pool_to_tokens(alpha, tokens).cpu().numpy()
            footprint = (pool_to_tokens(alpha, tokens, mode='max') >= FOOTPRINT_ALPHA).cpu().numpy()

        saved = {'image': F.interpolate(s['image'][None], size=a.out_size, mode='area')[0].byte().numpy(),
                 'label': s['label'][0].numpy().astype(np.uint8), 'clean_pred': clean_pred.cpu().numpy(),
                 'click': s['click'].numpy(), 'layers': np.array(layer_names)}
        if alpha_tokens is not None:
            saved['spot_alpha_tokens'] = alpha_tokens.astype(np.float16)
        for condition in conditions:
            degraded01, _ = degrader.apply(condition)
            logits_d, neck_d, blocks_d = predict(net, (degraded01 * 255)[None], click[None], a.out_size,
                                                 a.multimask_output, block_outputs=True)
            deg_layers = [b.float() for b in blocks_d] + [neck_d.permute(0, 2, 3, 1).float()]
            pred_d = logits_d[0, 0] > 0
            changed = pool_to_tokens((pred_d != clean_pred).float(), tokens).cpu().numpy()
            newly_wrong = pool_to_tokens(((pred_d != (label[0] > 0.5)) & (clean_pred == (label[0] > 0.5))).float(),
                                         tokens).cpu().numpy()
            maps = np.stack([token_change(c, d).cpu().numpy() for c, d in zip(clean_layers, deg_layers)])
            tag = condition.name
            saved[f'{tag}/change'] = maps.astype(np.float16)
            saved[f'{tag}/pred'] = pred_d.cpu().numpy()
            saved[f'{tag}/image'] = F.interpolate((degraded01 * 255)[None].cpu(), size=a.out_size,
                                                  mode='area')[0].byte().numpy()
            image_ssim = float(ssim(image01, degraded01))
            for name, change in zip(layer_names, maps):
                total = change.sum()
                row = {'image': s['name'], 'index': s['index'], 'condition': tag, 'kind': condition.kind,
                       'param': condition.param, 'level': level, 'ssim': image_ssim, 'layer': name,
                       'mean_change': float(change.mean()), 'max_change': float(change.max()),
                       'top1_share': top_share(change, 0.01), 'top5_share': top_share(change, 0.05),
                       'pred_changed_frac': float(changed.mean()), 'new_error_frac': float(newly_wrong.mean())}
                if footprint is not None:
                    row['footprint_tokens'] = int(footprint.sum())
                    row['outside_footprint_share'] = (float(change[~footprint].sum() / total)
                                                      if total > 0 and footprint.any() else float('nan'))
                for target_name, target in (('changed', changed), ('new_error', newly_wrong)):
                    row[f'spearman_{target_name}'] = (float(spearmanr(change.ravel(), target.ravel())[0])
                                                      if target.std() > 0 else float('nan'))
                rows.append(row)
        np.savez_compressed(os.path.join(a.out_dir, f"{s['name']}.npz"), **saved)

    write_csv(rows, os.path.join(a.out_dir, 'feature_stats.csv'))
    write_json({'level': level, 'conditions': [c.as_row() for c in conditions], 'images': chosen,
                'footprint_alpha': FOOTPRINT_ALPHA, 'levels_meta': meta,
                'global_attention_blocks': [j for j, blk in enumerate(net.image_encoder.blocks) if blk.window_size == 0]},
               os.path.join(a.out_dir, 'feature_meta.json'))


if __name__ == '__main__':
    main()
