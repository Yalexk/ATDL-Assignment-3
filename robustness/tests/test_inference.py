import numpy as np
import pytest
import torch

from robustness.inference import THRESHOLDS, segmentation_metrics


def reference_eval_seg(pred, true_mask_p, threshold):
    """utils.eval_seg (single-channel branch) with utils.iou / utils.dice_coeff inlined."""
    pred = torch.sigmoid(pred)
    eiou, edice = 0, 0
    for th in threshold:
        gt = (true_mask_p > th).float()
        vpred = (pred > th).float()
        p = vpred[:, 0].numpy().astype('int32')
        m = gt[:, 0].numpy().astype('int32')
        eiou += (((p & m).sum((1, 2)) + 1e-6) / ((p | m).sum((1, 2)) + 1e-6)).mean()
        dices = []
        for a, b in zip(vpred[:, 0], gt[:, 0]):
            inter = torch.dot(a.reshape(-1), b.reshape(-1))
            dices.append(((2 * inter + 1e-4) / (a.sum() + b.sum() + 1e-4)).item())
        edice += sum(dices) / len(dices)
    return eiou / len(threshold), edice / len(threshold)


def test_threshold_averaged_metrics_match_repository():
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(5, 1, 32, 32, generator=gen) * 3
    labels = (torch.rand(5, 1, 32, 32, generator=gen) > 0.6).float()
    labels[2] = 0  # empty mask edge case
    ours = segmentation_metrics(logits, labels)
    ref_iou, ref_dice = reference_eval_seg(logits, labels, THRESHOLDS)
    assert ours['iou_thavg'].mean() == pytest.approx(ref_iou, abs=1e-6)
    assert ours['dice_thavg'].mean() == pytest.approx(ref_dice, abs=1e-6)
    ref_iou05, ref_dice05 = reference_eval_seg(logits, labels, (0.5,))
    assert ours['dice05'].mean() == pytest.approx(ref_dice05, abs=1e-6)
    assert ours['iou05'].mean() == pytest.approx(ref_iou05, abs=1e-6)


def test_perfect_prediction_scores_one():
    labels = torch.zeros(1, 1, 8, 8)
    labels[..., 2:6, 2:6] = 1
    metrics = segmentation_metrics(labels * 20 - 10, labels)
    for value in metrics.values():
        assert value[0] == pytest.approx(1.0)
