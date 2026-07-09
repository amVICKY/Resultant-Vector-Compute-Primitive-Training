"""
evaluation/segmentation.py
--------------------------
Score a pretrained segmentation model (FCN-ResNet50) on Pascal VOC 2012 val and report
mean IoU, pixel accuracy, and macro precision.

Metrics are accumulated from a 21x21 confusion matrix over all valid pixels (the 255
"void" label is ignored). Working from the confusion matrix keeps every metric
consistent and lets us aggregate over the whole set in one pass.
"""

from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def _confusion(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Accumulate a [C, C] confusion matrix (rows = ground truth, cols = prediction)."""
    valid = (target >= 0) & (target < num_classes)  # drops 255 void pixels
    idx = num_classes * target[valid] + pred[valid]
    cm = torch.bincount(idx, minlength=num_classes ** 2)
    return cm.reshape(num_classes, num_classes)


@torch.no_grad()
def evaluate_segmentation(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int = 21,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Returns {"mean_iou", "pixel_acc", "precision", "n"}.

    mean_iou   : mean over classes of TP / (TP + FP + FN)
    pixel_acc  : fraction of valid pixels classified correctly
    precision  : macro mean over classes of TP / (TP + FP)
    Classes absent from both prediction and ground truth are skipped in the means.
    """
    model = model.to(device).eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    n_imgs = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        out = model(imgs)["out"]                       # [B, C, H, W]
        pred = out.argmax(dim=1)                       # [B, H, W]
        cm += _confusion(pred.flatten(), masks.flatten(), num_classes).to(cm.device)
        n_imgs += imgs.shape[0]

    cm = cm.double()
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    denom_iou = tp + fp + fn
    present = denom_iou > 0                             # class seen somewhere
    iou = tp[present] / denom_iou[present]
    mean_iou = float(iou.mean().item())

    pixel_acc = float((tp.sum() / cm.sum()).item())

    denom_prec = tp + fp
    pp = denom_prec > 0
    precision = float((tp[pp] / denom_prec[pp]).mean().item())

    return {"mean_iou": mean_iou, "pixel_acc": pixel_acc,
            "precision": precision, "n": int(n_imgs)}
