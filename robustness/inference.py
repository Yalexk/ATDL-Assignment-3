"""Forward pass and metrics matching function.validation_sam / utils.eval_seg for one-click prompts."""
import torch
import torch.nn.functional as F

THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)  # utils.eval_seg averages over these


@torch.no_grad()
def predict(net, images255, clicks, out_size, multimask_output=1, block_outputs=False):
    """Mask logits (B, 1, out_size, out_size) for images in [0, 255] and one positive click (x, y) each.

    With block_outputs=True also returns the image embedding (B, 256, 64, 64) and the list of
    ViT block outputs (each (B, 64, 64, C)) for feature-change analysis.
    """
    blocks = []
    hooks = []
    if block_outputs:
        hooks = [blk.register_forward_hook(lambda _m, _i, out: blocks.append(out))
                 for blk in net.image_encoder.blocks]
    try:
        embedding = net.image_encoder(net.preprocess(images255))
    finally:
        for hook in hooks:
            hook.remove()
    coords = clicks.to(dtype=torch.float, device=embedding.device).view(-1, 1, 2)
    labels = torch.ones(coords.shape[0], 1, dtype=torch.int, device=embedding.device)
    sparse, dense = net.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
    logits, _ = net.mask_decoder(
        image_embeddings=embedding,
        image_pe=net.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=(multimask_output > 1),
    )
    logits = F.interpolate(logits, size=(out_size, out_size), mode='bilinear', align_corners=False)
    if block_outputs:
        return logits, embedding, blocks
    return logits


def dice_iou(pred, gt):
    """Per-image Dice and IoU of two binary maps (B, ...)."""
    pred = pred.flatten(1).float()
    gt = gt.flatten(1).float()
    inter = (pred * gt).sum(1)
    dice = (2 * inter + 1e-4) / (pred.sum(1) + gt.sum(1) + 1e-4)            # utils.DiceCoeff
    iou = (inter + 1e-6) / (((pred + gt) > 0).float().sum(1) + 1e-6)         # utils.iou
    return dice, iou


@torch.no_grad()
def segmentation_metrics(logits, labels):
    """Per-image Dice/IoU at threshold 0.5 and averaged over THRESHOLDS (the repository's reported metric)."""
    prob = torch.sigmoid(logits[:, :1])
    labels = labels[:, :1]
    out = {}
    dice_sum = torch.zeros(prob.shape[0], dtype=torch.float64, device=prob.device)
    iou_sum = torch.zeros_like(dice_sum)
    for th in THRESHOLDS:
        dice, iou = dice_iou(prob > th, labels > th)
        dice_sum += dice.double()
        iou_sum += iou.double()
        if th == 0.5:
            out['dice05'], out['iou05'] = dice.double(), iou.double()
    out['dice_thavg'] = dice_sum / len(THRESHOLDS)
    out['iou_thavg'] = iou_sum / len(THRESHOLDS)
    return {k: v.cpu().numpy() for k, v in out.items()}
