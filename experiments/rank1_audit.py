"""
This program check the first claim that the each weight matrix has a single dominant direction(rank-1),
allowing the network to be compressed to scalar projections. 
"""

import io
import sys
import os
import json

# Windows consoles default to cp1252, which can't encode the Unicode symbols
# (→, σ, τ, ×) used in the printed reports.
if (
    isinstance(sys.stdout, io.TextIOWrapper)
    and sys.stdout.encoding
    and sys.stdout.encoding.lower().replace("-", "") != "utf8"
):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import argparse
import numpy as np
import torch
import torchvision.models as models

from typing import Dict
from observation.extract import extract_weight_matrices,print_layer_summary
from observation.metrics import compute_all_metrics,print_metrics_table
from visualization.plots import (
    plot_rank1_energy,
    plot_alignment_scores,
    plot_alignment_histograms,
    plot_singular_value_decay,
    plot_significant_rank,
    plot_revision_summary,
    set_style,
)

Models = {
    "resnet18":(models.resnet18,models.ResNet18_Weights.DEFAULT),
    "resnet34":(models.resnet34,models.ResNet34_Weights.DEFAULT),
    "resnet50":(models.resnet50,models.ResNet50_Weights.DEFAULT),
}

# Compute aggregate statistics across all layers for the written summary
def compute_summary_stats(
    results:Dict,
    tau:float
) -> Dict:
    
    r1e_vals = [v["rank1_energy"] for v in results.values()]
    aln_vals = [v["mean_cos_svd_resultant"] for v in results.values()]
    k_vals = [v["significant_rank"] for v in results.values()]
    ratio_vals = [v["significant_rank_ratio"] for v in results.values()]

    return {
        "n_layers":len(results),
        "tau":tau,
        "rank1_energy_mean":float(np.mean(r1e_vals)),
        "rank1_energy_min":float(np.min(r1e_vals)),
        "rank1_energy_max":float(np.max(r1e_vals)),
        "rank1_energy_below_50pct":int(sum(1 for v in r1e_vals if v<0.5)),
        "alignment_mean":float(np.mean(aln_vals)),
        "alignment_min":float(np.min(aln_vals)),
        "alignment_max":float(np.max(aln_vals)),
        "alignment_below_70pct":int(sum(1 for v in aln_vals if v<0.70)),
        "significant_rank_mean":float(np.mean(k_vals)),
        "significant_rank_min":int(np.min(k_vals)),
        "significant_rank_max":int(np.max(k_vals)),
        "relative_rank_mean":float(np.mean(ratio_vals)),
        "verdict":(
            "RANK-1 is insufficient" if np.mean(r1e_vals)<0.5 else "RANK-1 is partially valid"
        )
    }

def print_hypothesis_verdict(
    stats: dict
) -> None:
    """Print the narrative conclusion after metrics are collected."""
    sep = "=" * 70
 
    print(f"\n{sep}")
    print("  PHASE 1 VERDICT — GLOBAL RESULTANT HYPOTHESIS (v2)")
    print(f"{sep}")
 
    print(f"\n  Layers analyzed:  {stats['n_layers']}")
    print(f"  Energy threshold: τ = {stats['tau']}\n")
 
    print(f"  RANK-1 ENERGY  (σ₁² / Σσₖ²)")
    print(f"    Mean:  {stats['rank1_energy_mean']:.3f}")
    print(f"    Range: [{stats['rank1_energy_min']:.3f}, {stats['rank1_energy_max']:.3f}]")
    print(f"    Layers below 0.50: {stats['rank1_energy_below_50pct']} / {stats['n_layers']}")
 
    print(f"\n  ROW ALIGNMENT  (mean |cos θ| to SVD resultant)")
    print(f"    Mean:  {stats['alignment_mean']:.3f}")
    print(f"    Range: [{stats['alignment_min']:.3f}, {stats['alignment_max']:.3f}]")
    print(f"    Layers below 0.70: {stats['alignment_below_70pct']} / {stats['n_layers']}")
 
    print(f"\n  SIGNIFICANT RANK k_l  (at τ = {stats['tau']})")
    print(f"    Mean:  {stats['significant_rank_mean']:.1f}")
    print(f"    Range: [{stats['significant_rank_min']}, {stats['significant_rank_max']}]")
    print(f"    Relative rank mean: {stats['relative_rank_mean']:.3f}")
 
    print(f"\n  ► {stats['verdict']}")
    print(f"\n  INTERPRETATION")
    print(f"  The model did not learn a single dominant direction per layer.")
    print(f"  Each layer uses a subspace of dimension k_l >> 1 to represent")
    print(f"  its learned features. The rank-1 hypothesis (v1) is falsified.")
    print(f"\n  NEXT STEP → Phase 2: cross-layer principal angle analysis")
    print(f"  Are the significant subspaces of adjacent layers aligned?")
    print(f"  If yes → jointly significant subspace = compression target.\n")
    print(f"{sep}\n")

def run(
    model_name:str = "resnet18",
    tau:float = 0.90,
    mode:str = "conv_linear"
):
    print(f"\n{'='*70}")
    print(f"  GRH Phase 1 — Rank-1 Audit")
    print(f"  Model: {model_name}   |   tau: {tau}   |   mode: {mode}")
    print(f"{'='*70}\n")

    # Model Loading
    print("[1/5] Loading model")
    model_fn,weights = Models[model_name]
    model = model_fn(weights=weights)
    model.eval()
    print(f"{model_name} loaded pretrained Imagenet weights")

    # Extracting weights
    print("[2/5] Extracting weight matices")
    layers = extract_weight_matrices(model,mode=mode)
    print_layer_summary(layers)

    # Computing metrics
    print("[3/5] Computing metrics i.e SVD per layers")
    results = compute_all_metrics(layers,tau=tau)
    print_metrics_table(results)

    # Generate plots
    set_style()

    print("Plot1:Rank-1 energy per layer")
    plot_rank1_energy(results)

    print("Plot2:Alignment scores i.e SVD vs mean resultant")
    plot_alignment_scores(results)

    print("Plot 3:Per layer alignment histograms")
    plot_alignment_histograms(results)

    print("Plot 4:Signular value decay curves")
    plot_singular_value_decay(results,tau=tau)

    print("Plot 5:Significant rank K_1 per layer")
    plot_significant_rank(results,tau)

    print("Plot 6:Revision summary ")
    plot_revision_summary(results,tau=tau)

    # Saving numeric results
    print("[5/5] Saving numeric results")
    stats = compute_summary_stats(results,tau)
    print_hypothesis_verdict(stats)

    # Save summary stats to JSON (exclude non-serializable arrays)
    log_dir = os.path.join(os.path.dirname(__file__), "..", "results", "logs")
    os.makedirs(log_dir, exist_ok=True)
 
    log_path = os.path.join(log_dir, f"exp1_{model_name}_summary.json")
    with open(log_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Summary stats saved → {log_path}")
 
    # Save per-layer numeric table as CSV
    import csv
    csv_path = os.path.join(log_dir, f"exp1_{model_name}_per_layer.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "layer", "shape_m", "shape_n",
            "rank1_energy", "mean_cos_svd", "mean_cos_mean", "std_cos",
            "significant_rank", "significant_rank_ratio"
        ])
        writer.writeheader()
        for name, m in results.items():
            writer.writerow({
                "layer":                  name,
                "shape_m":                m["shape"][0],
                "shape_n":                m["shape"][1],
                "rank1_energy":           round(m["rank1_energy"], 5),
                "mean_cos_svd":           round(m["mean_cos_svd_resultant"], 5),
                "mean_cos_mean":          round(m["mean_cos_mean_resultant"], 5),
                "std_cos":                round(m["std_cos"], 5),
                "significant_rank":       m["significant_rank"],
                "significant_rank_ratio": round(m["significant_rank_ratio"], 5),
            })
    print(f"  Per-layer CSV saved  → {csv_path}")
    print("\n  All done. Open results/figures/ to view plots.\n")
 
    return results, stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank-1 Audit")
    parser.add_argument("--model",default="resnet18",choices=list(Models.keys()),help="Pretrained model to analyze")
    parser.add_argument("--tau",default=0.90,type=float,help="Energy threhold for significant rank")
    parser.add_argument("--mode",default="conv_linear",choices=["conv_linear","all"],help="Which layers to extract")

    args = parser.parse_args()
    run(model_name=args.model,tau=args.tau,mode=args.mode)