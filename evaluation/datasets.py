"""
evaluation/datasets.py
----------------------
Download / prepare the evaluation datasets and build ready-to-use DataLoaders.

Datasets
  - Imagenette (10-class subset of ImageNet). Lets us score an ImageNet-pretrained
    classifier without the full (non-downloadable) ImageNet val set. The 10 folders
    are real ImageNet WNIDs, so their labels map onto the model's 1000-way output.
  - Pascal VOC 2012 segmentation val. Ground-truth masks are required for IoU;
    torchvision's FCN/DeepLab weights are trained on this VOC label set (21 classes).

Run directly to fetch + extract both into ./data:
    python -m evaluation.datasets
"""

import os
from typing import List, Tuple

import torch
import torchvision as tv
from torch.utils.data import DataLoader

# Repo-root/data — keep everything the experiments read in one place.
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")

# The 10 Imagenette WNIDs, in the sorted order torchvision assigns labels 0..9,
# paired with their index in the 1000-way ImageNet classifier output. This lets us
# turn a model's full logit vector into a 10-way decision for these classes.
IMAGENETTE_WNIDS: List[str] = [
    "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
    "n03394916", "n03417042", "n03425413", "n03445777", "n03888257",
]
IMAGENETTE_TO_IMAGENET: List[int] = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]


def prepare_imagenette(root: str = DATA_ROOT) -> str:
    """Download + extract Imagenette val split (320px). Returns the data root."""
    tv.datasets.Imagenette(root, split="val", size="320px", download=True)
    return root


def prepare_voc(root: str = DATA_ROOT) -> str:
    """Download + extract Pascal VOC 2012 segmentation val split. Returns the data root."""
    tv.datasets.VOCSegmentation(root, year="2012", image_set="val", download=True)
    return root


def imagenette_loader(
    root: str = DATA_ROOT,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, List[int]]:
    """
    Build a val DataLoader for Imagenette using the ImageNet classifier transforms.

    Returns:
        loader          : yields (image, label_in_0..9)
        class_idx       : ImageNet output columns for the 10 classes, so predictions
                          can be restricted to a 10-way argmax.
    """
    # Reuse the exact preprocessing the pretrained weights expect.
    transform = tv.models.ResNet18_Weights.DEFAULT.transforms()
    ds = tv.datasets.Imagenette(
        root, split="val", size="320px", download=False, transform=transform
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, IMAGENETTE_TO_IMAGENET


def voc_loader(
    root: str = DATA_ROOT,
    batch_size: int = 8,
    num_workers: int = 0,
    eval_size: int = 480,
):
    """
    Build a val DataLoader for VOC2012 segmentation. Returns (loader, num_classes=21).

    Image and mask are resized *jointly* to a fixed square (bilinear for the image,
    nearest for the mask). Fixing the size keeps predictions and ground-truth on the
    same grid — required for a valid IoU — and lets images batch. Absolute mIoU will
    be a touch below the published number, but this study only needs the *relative*
    degradation as energy is dropped, so a consistent grid is what matters.
    """
    import numpy as np
    import torchvision.transforms.functional as TF

    # ImageNet normalization — matches how the FCN/DeepLab backbones were trained.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    class VOCJointResize(tv.datasets.VOCSegmentation):
        def __getitem__(self, index):
            img, mask = super().__getitem__(index)  # PIL image, PIL 'P' mask
            img = TF.resize(img, [eval_size, eval_size], interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [eval_size, eval_size], interpolation=TF.InterpolationMode.NEAREST)
            img = TF.normalize(TF.to_tensor(img), mean, std)
            # VOC masks: 0..20 = classes, 255 = void/ignore (handled downstream).
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)
            return img, mask

    ds = VOCJointResize(root, year="2012", image_set="val", download=False)

    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        masks = torch.stack([b[1] for b in batch])
        return imgs, masks

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate,
    )
    return loader, 21


if __name__ == "__main__":
    os.makedirs(DATA_ROOT, exist_ok=True)
    print(f"[datasets] data root: {os.path.abspath(DATA_ROOT)}")

    print("[datasets] preparing Imagenette (val, 320px, ~342 MB)...")
    prepare_imagenette()
    print("[datasets] Imagenette ready.")

    print("[datasets] preparing Pascal VOC 2012 seg val (~2 GB)...")
    prepare_voc()
    print("[datasets] VOC ready.")

    print("[datasets] DONE.")
