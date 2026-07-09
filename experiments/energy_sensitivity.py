"""
experiments/energy_sensitivity.py
---------------------------------
Relate weight-matrix ENERGY to task METRICS.

Procedure (per task):
  1. Load a pretrained model that already achieves good metrics.
  2. For each retained-energy level tau in the sweep, low-rank truncate every
     conv/linear weight to keep only tau of its Frobenius energy (Eckart-Young),
     re-insert the truncated weights, and evaluate.
  3. Record the metric(s) against the actual global retained energy.

The resulting curve answers the question directly: how much energy can be dropped
before accuracy / precision / IoU fall off, and how steep is the fall.

  Classification (Imagenette, ResNet18) -> accuracy, precision
  Segmentation   (VOC2012,  FCN-Res50)  -> mean IoU, pixel accuracy, precision

Run:
  python -m experiments.energy_sensitivity --task both
  python -m experiments.energy_sensitivity --task classification --limit 20
"""

import argparse
import csv
import io
import os
import sys
from typing import Dict, List

# Windows consoles default to cp1252 and choke on the unicode used in reports.
if (
    isinstance(sys.stdout, io.TextIOWrapper)
    and sys.stdout.encoding
    and sys.stdout.encoding.lower().replace("-", "") != "utf8"
):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torchvision as tv

from observation.truncate import truncate_model, global_retained_energy
from evaluation.datasets import imagenette_loader, voc_loader
from evaluation.classification import evaluate_classification
from evaluation.segmentation import evaluate_segmentation

DEFAULT_TAUS: List[float] = [1.0, 0.999, 0.99, 0.98, 0.95, 0.90, 0.80, 0.70, 0.50]
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "logs")


def _device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available — falling back to CPU (segmentation will be slow).")
        return "cpu"
    return requested


def run_classification_sweep(taus, device, batch_size=64) -> List[Dict]:
    print("\n=== CLASSIFICATION SWEEP (Imagenette / ResNet18) ===")
    weights = tv.models.ResNet18_Weights.DEFAULT
    base = tv.models.resnet18(weights=weights)
    loader, class_idx = imagenette_loader(batch_size=batch_size)

    rows: List[Dict] = []
    for tau in taus:
        model_t, info = truncate_model(base, tau=tau, mode="conv_linear")
        ge = global_retained_energy(info)
        m = evaluate_classification(model_t, loader, class_idx, device=device)
        row = {"tau": tau, "retained_energy": round(ge, 5),
               "accuracy": round(m["accuracy"], 5), "precision": round(m["precision"], 5)}
        rows.append(row)
        print(f"  tau={tau:<6} retained={ge:6.4f}  acc={m['accuracy']:.4f}  prec={m['precision']:.4f}")
    return rows


def run_segmentation_sweep(taus, device, batch_size=8) -> List[Dict]:
    print("\n=== SEGMENTATION SWEEP (VOC2012 / FCN-ResNet50) ===")
    weights = tv.models.segmentation.FCN_ResNet50_Weights.DEFAULT
    base = tv.models.segmentation.fcn_resnet50(weights=weights)
    loader, ncls = voc_loader(batch_size=batch_size)

    rows: List[Dict] = []
    for tau in taus:
        model_t, info = truncate_model(base, tau=tau, mode="conv_linear")
        ge = global_retained_energy(info)
        m = evaluate_segmentation(model_t, loader, num_classes=ncls, device=device)
        row = {"tau": tau, "retained_energy": round(ge, 5),
               "mean_iou": round(m["mean_iou"], 5), "pixel_acc": round(m["pixel_acc"], 5),
               "precision": round(m["precision"], 5)}
        rows.append(row)
        print(f"  tau={tau:<6} retained={ge:6.4f}  mIoU={m['mean_iou']:.4f}  "
              f"pixAcc={m['pixel_acc']:.4f}  prec={m['precision']:.4f}")
    return rows


def _save_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved -> {path}")


def _plot(cls_rows: List[Dict], seg_rows: List[Dict]) -> None:
    import matplotlib.pyplot as plt
    from visualization.plots import set_style, COLORS, SAVE_DIR
    set_style()

    n_panels = int(bool(cls_rows)) + int(bool(seg_rows))
    if n_panels == 0:
        return
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 4.6), squeeze=False)
    axes = axes[0]
    ax_i = 0

    if cls_rows:
        ax = axes[ax_i]; ax_i += 1
        x = [r["retained_energy"] for r in cls_rows]
        ax.plot(x, [r["accuracy"] for r in cls_rows], "o-", color=COLORS["primary"], label="accuracy")
        ax.plot(x, [r["precision"] for r in cls_rows], "s--", color=COLORS["accent"], label="precision")
        ax.set_title("Classification — Imagenette / ResNet18")
        ax.set_xlabel("retained energy (mean over layers)")
        ax.set_ylabel("metric")
        ax.invert_xaxis()  # energy dropped left→right
        ax.set_ylim(0, 1.02)
        ax.legend()

    if seg_rows:
        ax = axes[ax_i]; ax_i += 1
        x = [r["retained_energy"] for r in seg_rows]
        ax.plot(x, [r["mean_iou"] for r in seg_rows], "o-", color=COLORS["primary"], label="mean IoU")
        ax.plot(x, [r["pixel_acc"] for r in seg_rows], "^-", color=COLORS["ok"], label="pixel acc")
        ax.plot(x, [r["precision"] for r in seg_rows], "s--", color=COLORS["accent"], label="precision")
        ax.set_title("Segmentation — VOC2012 / FCN-ResNet50")
        ax.set_xlabel("retained energy (mean over layers)")
        ax.set_ylabel("metric")
        ax.invert_xaxis()
        ax.set_ylim(0, 1.02)
        ax.legend()

    fig.suptitle("Energy dropped from weights vs task performance", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(SAVE_DIR, "plot7_energy_vs_metrics.png")
    os.makedirs(SAVE_DIR, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  figure -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Energy vs metrics sensitivity sweep")
    parser.add_argument("--task", default="both", choices=["both", "classification", "segmentation"])
    parser.add_argument("--taus", type=float, nargs="+", default=DEFAULT_TAUS,
                        help="Retained-energy levels to sweep (1.0 = baseline)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--cls-batch", type=int, default=64)
    parser.add_argument("--seg-batch", type=int, default=8)
    args = parser.parse_args()

    device = _device(args.device)
    taus = sorted(set(args.taus), reverse=True)  # start at baseline, drop energy downward

    cls_rows, seg_rows = [], []
    if args.task in ("both", "classification"):
        cls_rows = run_classification_sweep(taus, device, batch_size=args.cls_batch)
        _save_csv(cls_rows, os.path.join(LOG_DIR, "energy_sensitivity_classification.csv"))
    if args.task in ("both", "segmentation"):
        seg_rows = run_segmentation_sweep(taus, device, batch_size=args.seg_batch)
        _save_csv(seg_rows, os.path.join(LOG_DIR, "energy_sensitivity_segmentation.csv"))

    print("\n[plot] building energy-vs-metrics figure...")
    _plot(cls_rows, seg_rows)
    print("\nDone. See results/logs/*.csv and results/figures/plot7_energy_vs_metrics.png\n")


if __name__ == "__main__":
    main()
