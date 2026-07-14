"""
experiments/model_analysis.py
-----------------------------
Runs the Global Shared Basis test (Phase 2 / `global_basis_check.py`) across a
range of torchvision models — CNNs, a Vision Transformer, and a light Swin
Transformer — to see whether the GRH shared-basis hypothesis holds beyond a
single ResNet18.

For each model it:
    1. Loads ImageNet-pretrained weights.
    2. Extracts conv/linear weight matrices and computes k_l per layer.
    3. Groups layers by input dimension and builds a global basis per group.
    4. Measures per-layer reconstruction ratio (global basis / private SVD).
    5. Runs the nesting test on the largest group.

Outputs (under results/model_analysis/):
    <model>/kl_profile.png
    <model>/reconstruction_ratio.png
    <model>/nesting_heatmap.png
    <model>/metrics.json          full per-layer metrics + ratios
    <model>/ratios.csv            flat per-layer table
    summary/cross_model_comparison.png
    summary/summary.json
    summary/summary.csv
    model_analysis.log            full run log
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import json
import logging
import traceback

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import subspace_angles

import torchvision.models as tv_models

from observation.extract import extract_weight_matrices
from observation.metrics import compute_all_metrics
from visualization.plots import set_style, COLORS, _short_name

# Reuse the Phase-2 machinery instead of re-implementing it.
from experiments.global_basis_check import (
    group_layers_by_input_dim,
    build_global_basis,
    reconstruction_ratio,
    plot_kl_profile,
    plot_reconstruction_ratios,
    plot_nesting_heatmap,
)

# ── Config ──────────────────────────────────────────────────────────────────

# torchvision model names resolved via models.get_model(name, weights="DEFAULT").
# Mix of CNNs + a ViT + Swin-Tiny (the lightest transformer in torchvision).
MODELS = [
    "resnet18",
    "resnet50",
    "vgg16",
    "densenet121",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "vit_b_16",
    "swin_t",
]

TAU = 0.90  # energy threshold for significant rank

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "model_analysis")

# Thresholds for the "shared basis works" verdict (mirrors global_basis_check).
RATIO_GOOD = 1.15
RATIO_WARN = 1.50

logger = logging.getLogger("model_analysis")


# ── Logging setup ───────────────────────────────────────────────────────────

def setup_logging(log_path):
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def _add_model_logfile(log_path):
    """Attach a dedicated file handler so this model's run is captured on its own."""
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return fh


# ── JSON helper (numpy → native) ────────────────────────────────────────────

def _to_native(obj):
    """Recursively convert numpy scalars/arrays to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


# ── Per-model analysis ──────────────────────────────────────────────────────

def analyze_model(name):
    """Run the full Phase-2 pipeline on one model. Returns a summary dict."""
    logger.info("=" * 70)
    logger.info("MODEL: %s", name)
    logger.info("=" * 70)

    model_dir = os.path.join(OUT_DIR, name)
    os.makedirs(model_dir, exist_ok=True)

    # Dedicated per-model log file inside the model's folder.
    model_log = _add_model_logfile(os.path.join(model_dir, f"{name}.log"))
    try:
        return _analyze_model_body(name, model_dir)
    finally:
        logger.removeHandler(model_log)
        model_log.close()


def _analyze_model_body(name, model_dir):
    # 1. Load pretrained model
    logger.info("[1/5] Loading %s (pretrained)...", name)
    model = tv_models.get_model(name, weights="DEFAULT")
    model.eval()

    # 2. Extract weights + per-layer metrics
    logger.info("[2/5] Extracting weights and computing k_l per layer...")
    layers = extract_weight_matrices(model)
    results = compute_all_metrics(layers, tau=TAU)
    logger.info("      %d weight matrices extracted", len(layers))

    fig = plot_kl_profile(results, save=False)
    fig.savefig(os.path.join(model_dir, "kl_profile.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Group layers by input dimension
    logger.info("[3/5] Grouping layers by input dimension...")
    groups = group_layers_by_input_dim(layers)
    logger.info("      %d shareable groups (input-dim shared by >1 layer)", len(groups))
    for d, g in sorted(groups.items()):
        logger.info("      d=%-5d: %d layers  -> %s", d, len(g), list(g.keys()))

    # 4. Global basis per group + reconstruction ratios
    logger.info("[4/5] Building global basis per group and measuring ratios...")
    ratio_results = {}
    for d, group in groups.items():
        logger.info("  Group d=%d:", d)
        k_max = max(results[n]["significant_rank"] for n in group)
        B = build_global_basis(group, k_max=k_max)
        for lname, W in group.items():
            k_l = results[lname]["significant_rank"]
            ratio = reconstruction_ratio(W, B, k_l)
            ratio_results[lname] = {"ratio": float(ratio), "k_l": int(k_l), "d": int(d)}
            status = "OK" if ratio < RATIO_GOOD else "WARN" if ratio < RATIO_WARN else "FAIL"
            logger.info("    %-4s %-38s k_l=%4d  ratio=%.4f", status, lname, k_l, ratio)

    if not ratio_results:
        logger.warning("      No shareable groups for %s — skipping ratio/nesting plots.", name)

    if ratio_results:
        fig = plot_reconstruction_ratios(ratio_results, save=False)
        fig.savefig(os.path.join(model_dir, "reconstruction_ratio.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 5. Nesting test within the largest group
    logger.info("[5/5] Testing nesting hypothesis on largest group...")
    nesting_results = {}
    if groups:
        largest_group_d = max(groups, key=lambda d: len(groups[d]))
        largest_group = groups[largest_group_d]
        group_names = list(largest_group.keys())

        layer_svd = {}
        for lname, W in largest_group.items():
            W_np = W.numpy()
            _, _, Vt = np.linalg.svd(W_np, full_matrices=False)
            k_l = results[lname]["significant_rank"]
            layer_svd[lname] = Vt[:k_l, :].T

        for i, n1 in enumerate(group_names):
            for j, n2 in enumerate(group_names):
                if j <= i:
                    continue
                angles = subspace_angles(layer_svd[n1], layer_svd[n2])
                nesting_results[(n1, n2)] = float(np.degrees(angles).mean())

        if nesting_results:
            fig = plot_nesting_heatmap(nesting_results, save=False)
            fig.savefig(os.path.join(model_dir, "nesting_heatmap.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    # ── Persist per-model logs (JSON + CSV) ─────────────────────────────
    _write_model_json(model_dir, name, results, ratio_results, nesting_results)
    _write_model_csv(model_dir, results, ratio_results)

    # ── Verdict ─────────────────────────────────────────────────────────
    ratios = [v["ratio"] for v in ratio_results.values()]
    if ratios:
        mean_r = float(np.mean(ratios))
        max_r = float(np.max(ratios))
        n_ok = int(sum(1 for r in ratios if r < RATIO_GOOD))
    else:
        mean_r = max_r = float("nan")
        n_ok = 0

    mean_angle = float(np.mean(list(nesting_results.values()))) if nesting_results else float("nan")

    if not ratios:
        verdict = "NO_GROUPS"
    elif mean_r < RATIO_GOOD:
        verdict = "SHARED_BASIS_WORKS"
    elif mean_r < RATIO_WARN:
        verdict = "PARTIAL"
    else:
        verdict = "INSUFFICIENT"

    logger.info("=" * 65)
    logger.info("  PHASE 2 VERDICT — %s", name)
    logger.info("=" * 65)
    logger.info("  Mean reconstruction ratio : %.4f", mean_r)
    logger.info("  Max reconstruction ratio  : %.4f", max_r)
    logger.info("  Layers well-served (< %.2f): %d / %d", RATIO_GOOD, n_ok, len(ratios))
    logger.info("  Mean nesting angle (deg)  : %.1f", mean_angle)
    if verdict == "SHARED_BASIS_WORKS":
        logger.info("  -> SHARED BASIS WORKS")
    elif verdict == "PARTIAL":
        logger.info("  -> SHARED BASIS PARTIALLY WORKS (some layers resist sharing)")
    elif verdict == "INSUFFICIENT":
        logger.info("  -> SHARED BASIS INSUFFICIENT (layers too diverse for one basis)")
    else:
        logger.info("  -> NO SHAREABLE GROUPS (no input-dim shared by >1 layer)")
    logger.info("=" * 65)

    return {
        "model": name,
        "n_layers": len(layers),
        "n_groups": len(groups),
        "n_shared_layers": len(ratios),
        "mean_ratio": mean_r,
        "max_ratio": max_r,
        "n_ok": n_ok,
        "frac_ok": (n_ok / len(ratios)) if ratios else float("nan"),
        "mean_nesting_angle_deg": mean_angle,
        "verdict": verdict,
    }


def _write_model_json(model_dir, name, results, ratio_results, nesting_results):
    """Full per-layer metrics + ratios (heavy arrays dropped)."""
    layers_out = {}
    for lname, m in results.items():
        entry = {
            "shape": list(m["shape"]),
            "rank1_energy": m["rank1_energy"],
            "mean_cos_svd_resultant": m["mean_cos_svd_resultant"],
            "mean_cos_mean_resultant": m["mean_cos_mean_resultant"],
            "std_cos": m["std_cos"],
            "significant_rank": m["significant_rank"],
            "significant_rank_ratio": m["significant_rank_ratio"],
        }
        if lname in ratio_results:
            entry["reconstruction_ratio"] = ratio_results[lname]["ratio"]
            entry["group_input_dim"] = ratio_results[lname]["d"]
        layers_out[lname] = entry

    payload = {
        "model": name,
        "tau": TAU,
        "layers": layers_out,
        "nesting_angles_deg": [
            {"layer_a": a, "layer_b": b, "mean_angle_deg": v}
            for (a, b), v in nesting_results.items()
        ],
    }
    path = os.path.join(model_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_native(payload), f, indent=2)


def _write_model_csv(model_dir, results, ratio_results):
    path = os.path.join(model_dir, "ratios.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "layer", "rows", "cols", "significant_rank", "significant_rank_ratio",
            "rank1_energy", "mean_cos_svd_resultant", "group_input_dim",
            "reconstruction_ratio",
        ])
        for lname, m in results.items():
            rr = ratio_results.get(lname, {})
            w.writerow([
                lname, m["shape"][0], m["shape"][1], m["significant_rank"],
                f"{m['significant_rank_ratio']:.6f}", f"{m['rank1_energy']:.6f}",
                f"{m['mean_cos_svd_resultant']:.6f}",
                rr.get("d", ""),
                f"{rr['ratio']:.6f}" if "ratio" in rr else "",
            ])


# ── Cross-model summary ─────────────────────────────────────────────────────

def plot_cross_model_comparison(summaries, save_path):
    """Bar chart of mean/max reconstruction ratio per model + fraction well-served."""
    set_style()
    summaries = [s for s in summaries if not np.isnan(s["mean_ratio"])]
    if not summaries:
        return
    names = [s["model"] for s in summaries]
    mean_r = [s["mean_ratio"] for s in summaries]
    max_r = [s["max_ratio"] for s in summaries]
    frac_ok = [s["frac_ok"] for s in summaries]
    short = [_short_name(n, 16) for n in names]
    x = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(9, len(names) * 1.1), 8))

    width = 0.38
    ax1.bar(x - width / 2, mean_r, width, label="mean ratio",
            color=COLORS["primary"], alpha=0.85, edgecolor="white")
    ax1.bar(x + width / 2, max_r, width, label="max ratio",
            color=COLORS["secondary"], alpha=0.7, edgecolor="white")
    ax1.axhline(RATIO_GOOD, color="#1F2937", linestyle="--", linewidth=1.0,
                label=f"{RATIO_GOOD} = acceptable")
    ax1.axhline(RATIO_WARN, color=COLORS["secondary"], linestyle=":", linewidth=1.0,
                label=f"{RATIO_WARN} = problem")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Reconstruction ratio (global / optimal)")
    ax1.set_title("Cross-Model Shared-Basis Quality\n"
                  "Lower = a single shared basis reconstructs layers well")
    ax1.legend(fontsize=8)

    colors = [COLORS["ok"] if f > 0.66 else COLORS["accent"] if f > 0.33
              else COLORS["secondary"] for f in frac_ok]
    ax2.bar(x, frac_ok, color=colors, alpha=0.85, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Fraction of layers well-served (< 1.15)")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Fraction of Layers Where Shared Basis Works")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → %s", save_path)


def write_summary(summaries, summary_dir):
    os.makedirs(summary_dir, exist_ok=True)

    with open(os.path.join(summary_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(_to_native(summaries), f, indent=2)

    csv_path = os.path.join(summary_dir, "summary.csv")
    fields = ["model", "n_layers", "n_groups", "n_shared_layers", "mean_ratio",
              "max_ratio", "n_ok", "frac_ok", "mean_nesting_angle_deg", "verdict"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            w.writerow({k: s.get(k, "") for k in fields})

    plot_cross_model_comparison(summaries, os.path.join(summary_dir, "cross_model_comparison.png"))


# ── Main ────────────────────────────────────────────────────────────────────

def run(model_names=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    setup_logging(os.path.join(OUT_DIR, "model_analysis.log"))

    model_names = model_names or MODELS
    logger.info("GRH Phase 2 — Cross-Model Shared Basis Test")
    logger.info("Models: %s", ", ".join(model_names))

    summaries = []
    for name in model_names:
        try:
            summaries.append(analyze_model(name))
        except Exception as e:  # keep the sweep alive if one model fails
            logger.error("FAILED on %s: %s", name, e)
            logger.debug(traceback.format_exc())
            summaries.append({
                "model": name, "n_layers": 0, "n_groups": 0, "n_shared_layers": 0,
                "mean_ratio": float("nan"), "max_ratio": float("nan"), "n_ok": 0,
                "frac_ok": float("nan"), "mean_nesting_angle_deg": float("nan"),
                "verdict": "ERROR",
            })

    summary_dir = os.path.join(OUT_DIR, "summary")
    write_summary(summaries, summary_dir)

    # Final console table
    logger.info("=" * 70)
    logger.info("CROSS-MODEL SUMMARY")
    logger.info("=" * 70)
    logger.info("%-22s %8s %8s %8s  %s", "model", "mean_r", "max_r", "frac_ok", "verdict")
    for s in summaries:
        logger.info("%-22s %8.4f %8.4f %8.2f  %s",
                    s["model"], s["mean_ratio"], s["max_ratio"], s["frac_ok"], s["verdict"])
    logger.info("Outputs written under: %s", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-model GRH shared-basis test")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of torchvision model names to run (default: full set)")
    args = parser.parse_args()
    run(args.models)
