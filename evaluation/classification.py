"""
evaluation/classification.py
----------------------------
Score an ImageNet-pretrained classifier on Imagenette and report top-1 accuracy and
macro precision. The model outputs 1000 logits; we restrict them to the 10 Imagenette
ImageNet columns and take a 10-way argmax, so the task is well defined on this subset.
"""

from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    class_idx: List[int],
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Returns {"accuracy", "precision", "n"} where precision is macro-averaged over the
    10 classes (unweighted mean of per-class precision).
    """
    from sklearn.metrics import precision_score

    model = model.to(device).eval()
    cols = torch.tensor(class_idx, device=device)

    all_pred: List[int] = []
    all_true: List[int] = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)                 # [B, 1000]
        sub = logits.index_select(1, cols)   # [B, 10] — only the Imagenette classes
        pred = sub.argmax(dim=1)             # 0..9
        all_pred.extend(pred.cpu().tolist())
        all_true.extend(labels.tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    acc = float((y_true == y_pred).mean())
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    return {"accuracy": acc, "precision": prec, "n": int(len(y_true))}
