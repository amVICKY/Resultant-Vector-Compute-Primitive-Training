import os
import sys
sys.path.insert(0,os.path.join(os.path.dirname(__file__),".."))

import numpy as np
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
from collections import defaultdict

from observation.extract import extract_weight_matrices
from observation.metrics import compute_all_metrics
from visualization.plots import set_style,COLORS,SAVE_DIR,_save,_short_name

# Group layers by the input dimension
def group_layers_by_input_dim(layers):
    groups = defaultdict(dict)
    for name,W in layers.items():
        d = W.shape[1]
        groups[d][name] = W
    return {d:g for d,g in groups.items() if len(g)>1}

# Try to build global basis (if exist) for a group
def build_global_basis(layer_group,k_max=None):
    matrices = list(layer_group.values())
    d = matrices[0].shape[1]

    normalized = []
    for W in matrices:
        W_np = W.numpy()
        frob = np.linalg.norm(W_np,'fro')
        if frob>1e-10:
            normalized.append(W_np/frob)
    
    W_stack = np.vstack(normalized)

    # SVD - right singular vector=global basis
    _,_,Vt = np.linalg.svd(W_stack,full_matrices=False)
    B = Vt.T
    if k_max is not None:
        B = B[:,:k_max]
    return B # [d,k_max]

# Measuring the quality of reconstruction
def optimal_reconstruction_error(W, k):
    W_np = W.numpy() if isinstance(W,torch.Tensor) else W
    U, S, Vt = np.linalg.svd(W_np,full_matrices=False)
    k = min(k,len(S))
    W_approx = (U[:,:k]*S[:k]) @ Vt[:k,:]
    return np.linalg.norm(W_np-W_approx,'fro')

def global_basis_reconstruction_error(W,B,k):
    W_np = W.numpy() if isinstance(W,torch.Tensor) else W
    k = min(k,B.shape[1])
    B_k = B[:,:k]
    A = W_np @ B_k
    W_approx = A @ B_k.T
    return np.linalg.norm(W_np - W_approx, 'fro')

def reconstruction_ratio(W,B,k):
    # 1.0 -> shared basis works well
    # 1.1 -> 10 % worse than optimal
    # 2.0 -> shared basis is quite bad
    opt = optimal_reconstruction_error(W, k)
    glob = global_basis_reconstruction_error(W,B,k)
    if opt < 1e-10:
        return 1.0 # layer is near-zero
    return glob/opt 

# Are early-layer subspaces actually contained inside deep-layer subspaces?
# If yes: using the deep-layer basis prefix serves everyone.
# Measured by: principal angles between early and deep significant subspaces.
def test_nesting(V_early,V_deep):
    from scipy.linalg import subspace_angles
    angles = subspace_angles(V_early,V_deep)
    return np.degrees(angles)

def plot_kl_profile(results, save=True):
    """
    k_l across layer depth — the 'staircase' showing progressive
    dimensionality expansion as depth increases.
    """
    set_style()
    names  = list(results.keys())
    k_vals = [results[n]["significant_rank"] for n in names]
    short  = [_short_name(n) for n in names]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.55), 4))
    bars = ax.bar(range(len(names)), k_vals,
                  color=COLORS["primary"], alpha=0.82, edgecolor="white")

    # Color by stage: early=green, mid=amber, deep=red
    stage_colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(names)))
    for bar, color in zip(bars, stage_colors):
        bar.set_facecolor(color)

    ax.plot(range(len(names)), k_vals, 'o--',
            color="#1F2937", linewidth=1.0, markersize=3, alpha=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(short, rotation=60, ha="right", fontsize=7.5)
    ax.set_ylabel("Significant rank  k_l  (τ=0.90)")
    ax.set_title(
        "Exp 2 — k_l Profile Across Depth\n"
        "Rising trend = network progressively activates more dimensions"
    )
    fig.tight_layout()
    if save:
        _save(fig, "exp2_kl_profile.png")
    return fig


def plot_reconstruction_ratios(ratio_results, save=True):
    """
    Per-layer ratio: global basis error / optimal error.
    Bars close to 1.0 → shared basis works. High bars → it doesn't.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(max(10, len(ratio_results) * 0.55), 5))

    names  = list(ratio_results.keys())
    ratios = [ratio_results[n]["ratio"] for n in names]
    k_vals = [ratio_results[n]["k_l"]   for n in names]
    short  = [_short_name(n)            for n in names]

    colors = [COLORS["ok"] if r < 1.15 else
              COLORS["accent"] if r < 1.5 else
              COLORS["secondary"] for r in ratios]

    ax.bar(range(len(names)), ratios, color=colors, alpha=0.82, edgecolor="white")
    ax.axhline(1.0,  color="#1F2937",          linestyle="--", linewidth=1.2,
               label="1.0 = perfect (same as private SVD)")
    ax.axhline(1.15, color=COLORS["accent"],   linestyle=":",  linewidth=1.0,
               label="1.15 = 15% worse (acceptable)")
    ax.axhline(1.50, color=COLORS["secondary"],linestyle=":",  linewidth=1.0,
               label="1.50 = 50% worse (problem)")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(short, rotation=60, ha="right", fontsize=7.5)
    ax.set_ylabel("Reconstruction ratio  (global / optimal)")
    ax.set_title(
        "Exp 2 — Global Basis Reconstruction Quality Per Layer\n"
        "Green < 1.15: shared basis works  |  Red > 1.5: layer resists sharing"
    )
    ax.legend(fontsize=8)
    ax.set_ylim(0.9, max(ratios) * 1.15)
    fig.tight_layout()
    if save:
        _save(fig, "exp2_reconstruction_ratio.png")
    return fig


def plot_nesting_heatmap(nesting_results, save=True):
    """
    Heatmap of mean principal angles between every pair of layers.
    Near-zero angles = early subspace nested inside deep subspace.
    """
    set_style()
    names = list(nesting_results.keys())
    n = len(names)
    matrix = np.zeros((n, n))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if (ni, nj) in nesting_results:
                matrix[i, j] = nesting_results[(ni, nj)]
            elif (nj, ni) in nesting_results:
                matrix[i, j] = nesting_results[(nj, ni)]

    short = [_short_name(nm, 16) for nm in names]
    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(6, n * 0.5)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=90, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=7)
    plt.colorbar(im, ax=ax, label="Mean principal angle (degrees)")
    ax.set_title(
        "Exp 2 — Nesting Test: Principal Angles Between Layer Subspaces\n"
        "Near 0° = early subspace contained in deep subspace (nesting holds)"
    )
    fig.tight_layout()
    if save:
        _save(fig, "exp2_nesting_heatmap.png")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────

def run():
    print("\n" + "="*65)
    print("  GRH Phase 2 — Global Shared Basis Test")
    print("="*65 + "\n")

    # Load model
    print("[1/5] Loading model...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()

    # Extract layers and compute k_l from Phase 1
    print("[2/5] Extracting weights and computing k_l per layer...")
    layers  = extract_weight_matrices(model)
    results = compute_all_metrics(layers, tau=0.90)

    # k_l profile plot
    plot_kl_profile(results)
    print("      → k_l profile plotted")

    # Group layers by input dimension
    print("[3/5] Grouping layers by input dimension...")
    groups = group_layers_by_input_dim(layers)
    for d, g in groups.items():
        print(f"      d={d:5d}: {len(g)} layers  → {list(g.keys())}")

    # Build global basis per group and compute reconstruction ratios
    print("[4/5] Building global basis per group and measuring ratios...")
    ratio_results = {}

    for d, group in groups.items():
        print(f"\n  Group d={d}:")
        k_max = max(results[n]["significant_rank"] for n in group)
        B = build_global_basis(group, k_max=k_max)

        for name, W in group.items():
            k_l   = results[name]["significant_rank"]
            ratio = reconstruction_ratio(W, B, k_l)
            ratio_results[name] = {"ratio": ratio, "k_l": k_l, "d": d}
            status = "✓" if ratio < 1.15 else "⚠" if ratio < 1.5 else "✗"
            print(f"    {status} {name:<38} k_l={k_l:4d}  ratio={ratio:.4f}")

    plot_reconstruction_ratios(ratio_results)

    # Nesting test — within the largest group
    print("[5/5] Testing nesting hypothesis...")
    largest_group_d = max(groups, key=lambda d: len(groups[d]))
    largest_group   = groups[largest_group_d]
    group_names     = list(largest_group.keys())

    # Get right singular vectors per layer
    layer_svd = {}
    for name, W in largest_group.items():
        W_np = W.numpy()
        _, _, Vt = np.linalg.svd(W_np, full_matrices=False)
        k_l = results[name]["significant_rank"]
        layer_svd[name] = Vt[:k_l, :].T   # [d, k_l]

    from scipy.linalg import subspace_angles
    nesting_results = {}
    for i, n1 in enumerate(group_names):
        for j, n2 in enumerate(group_names):
            if j <= i:
                continue
            V1, V2 = layer_svd[n1], layer_svd[n2]
            angles = subspace_angles(V1, V2)
            nesting_results[(n1, n2)] = float(np.degrees(angles).mean())

    plot_nesting_heatmap(nesting_results, save=True)

    # Final verdict
    ratios = [v["ratio"] for v in ratio_results.values()]
    mean_r = np.mean(ratios)
    max_r  = np.max(ratios)
    n_ok   = sum(1 for r in ratios if r < 1.15)

    print("\n" + "="*65)
    print("  PHASE 2 VERDICT")
    print("="*65)
    print(f"  Mean reconstruction ratio : {mean_r:.4f}")
    print(f"  Max reconstruction ratio  : {max_r:.4f}")
    print(f"  Layers well-served (< 1.15): {n_ok} / {len(ratios)}")
    if mean_r < 1.15:
        print("\n  ► SHARED BASIS WORKS")
        print("    Global basis reconstructs layers nearly as well as")
        print("    their own private SVD. Proceed to rotation analysis.")
    elif mean_r < 1.5:
        print("\n  ► SHARED BASIS PARTIALLY WORKS")
        print("    Some layers resist sharing. Check which ones (high ratio).")
        print("    Consider per-stage basis or higher k_max.")
    else:
        print("\n  ► SHARED BASIS INSUFFICIENT")
        print("    Layers are too diverse for a single shared basis.")
        print("    The hypothesis needs further revision.")
    print("="*65 + "\n")


if __name__ == "__main__":
    run()