"""Evaluate a model on clean and degraded copies of every image, one CSV row per (image, condition)."""
import math

import torch
from tqdm import tqdm

from .conditions import Condition, ImageDegrader
from .degradations import SpotConfig
from .inference import dice_iou, predict, segmentation_metrics
from .quality import psnr, ssim

CLEAN = Condition('clean')


@torch.no_grad()
def evaluate(net, loader, conditions, device, out_size, multimask_output=1, degrade_seed=0,
             spot_cfg=SpotConfig()):
    """Return a list of row dicts. The clean condition is always evaluated (and reported) first,
    because every degraded prediction is also compared with the clean prediction of the same image."""
    conditions = [CLEAN] + [c for c in conditions if c.kind != 'clean']
    split_id = loader.dataset.split_id
    rows = []
    for batch in tqdm(loader, desc='images', unit='batch'):
        images255 = batch['image'].to(device=device, dtype=torch.float32)
        images01 = images255 / 255
        labels = batch['label'].to(device)
        clicks = batch['click'].to(device=device, dtype=torch.float32)
        lesions = batch['lesion'].numpy()
        degraders = [ImageDegrader(images01[i], lesions[i], batch['click'][i].tolist(), split_id,
                                   int(batch['index'][i]), degrade_seed, spot_cfg)
                     for i in range(images01.shape[0])]
        clean_pred = None
        for condition in conditions:
            if condition.kind == 'clean':
                inputs255, infos = images255, [{}] * len(degraders)
                q_ssim = torch.ones(len(degraders), dtype=torch.float64)
                q_psnr = torch.full((len(degraders),), math.inf, dtype=torch.float64)
            else:
                outs = [d.apply(condition) for d in degraders]
                degraded01 = torch.stack([o[0] for o in outs])
                infos = [o[1] for o in outs]
                q_ssim, q_psnr = ssim(images01, degraded01).cpu(), psnr(images01, degraded01).cpu()
                inputs255 = degraded01 * 255
            logits = predict(net, inputs255, clicks, out_size, multimask_output)
            metrics = segmentation_metrics(logits, labels)
            pred = logits[:, :1] > 0
            if clean_pred is None:
                clean_pred = pred
            agreement = dice_iou(clean_pred, pred)[0].cpu()
            for i in range(len(degraders)):
                row = {
                    'image': batch['name'][i],
                    'index': int(batch['index'][i]),
                    'split_id': split_id,
                    **condition.as_row(),
                    'ssim': float(q_ssim[i]),
                    'psnr': float(q_psnr[i]),
                    **{k: float(v[i]) for k, v in metrics.items()},
                    'dice05_vs_clean': float(agreement[i]),
                    'pred_area_frac': float(pred[i].float().mean()),
                    'gt_area_frac': float(labels[i].float().mean()),
                    'click_x': float(batch['click'][i][0]),
                    'click_y': float(batch['click'][i][1]),
                }
                row.update(infos[i])
                rows.append(row)
    return rows
