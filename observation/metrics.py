import torch
import numpy as np
from typing import Dict,Tuple,Union

# Fraction of total Frobenius energy captured by the top singular value
def rank1_energy(
    W:torch.Tensor
) -> float:
    S = torch.linalg.svdvals(W)
    return (S[0]**2 / (S**2).sum()).item()

# Full cumulative energy profile of the singular value spectrum
def singular_value_energy(
    W:torch.Tensor    
) -> np.ndarray:
    S = torch.linalg.svdvals(W).numpy()
    energy = (S**2).cumsum() / (S**2).sum()
    return energy

# Minimum rank k of the top k values to capture "tau" fraction of energy
def significant_rank(
    W:torch.Tensor,
    tau:float=0.90
) -> int:
    energy = singular_value_energy(W)
    k = int(np.searchsorted(energy,tau))+1
    return min(k,W.shape[1])

# Measure how well the  weight rows align to a single dominant direction
def alignment_score(
    W:torch.Tensor
) -> Dict[str,Union[float,np.ndarray]]:
    r_mean = W.mean(dim=0)
    norm_mean = r_mean.norm()
    if norm_mean < 1e-10:
        cos_mean_vals = torch.zeros(W.shape[0])
    else:
        r_mean = r_mean / norm_mean
        W_norm = W / (W.norm(dim=1,keepdim=True) + 1e-8)
        cos_mean_vals = (W_norm @ r_mean).abs()
    
    try:
        _, _, Vt = torch.linalg.svd(W, full_matrices=False)
        r_svd = Vt[0]
        W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        cos_svd_vals = (W_norm @ r_svd).abs()
    except Exception:
        cos_svd_vals = torch.zeros(W.shape[0])
 
    return {
        "mean_cos_mean_resultant": cos_mean_vals.mean().item(),
        "mean_cos_svd_resultant":  cos_svd_vals.mean().item(),
        "std_cos":                 cos_svd_vals.std().item(),
        "cos_value":               cos_svd_vals.numpy(),
    }

# Run all metrics on every layer and return a unified results
def compute_all_metrics(
    layers:Dict[str,torch.Tensor],
    tau:float = 0.90
) ->Dict[str,Dict]:
    results = {}
    for name,W in layers.items():
        m,n = W.shape
        align = alignment_score(W)
        k = significant_rank(W,tau=tau)
        energy_curve = singular_value_energy(W)
        r1e  =rank1_energy(W)
        results[name] = {
            "shape":(m,n),
            "rank1_energy":r1e,
            "mean_cos_svd_resultant":align["mean_cos_svd_resultant"],
            "mean_cos_mean_resultant":align["mean_cos_mean_resultant"],
            "std_cos":align["std_cos"],
            "cos_values":align["cos_values"],
            "significant_rank":k,
            "significant_rank_ratio":k / min(m,n),
            "cumulative_energy":energy_curve,
        }
    return results

# Print a formatted table of per-layer metrics
def print_metrics_table(
    results:Dict[str,Dict]
)->None:
    header = (
        f"{'Layer':<42} | "
        f"{'Shape':<14} | "
        f"{'Rank1 E':>8} | "
        f"{'Align':>7} | "
        f"{'Std':>6} | "
        f"{'k@90%':>6} | "
        f"{'k/min(m,n)':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
 
    for name, m in results.items():
        shape_str = f"{m['shape'][0]}×{m['shape'][1]}"
        verdict = "ok" if m["rank1_energy"] > 0.5 else "✗ low"
        print(
            f"{name:<42} | "
            f"{shape_str:<14} | "
            f"{m['rank1_energy']:>7.3f}{verdict[1]} | "
            f"{m['mean_cos_svd_resultant']:>7.3f} | "
            f"{m['std_cos']:>6.3f} | "
            f"{m['significant_rank']:>6d} | "
            f"{m['significant_rank_ratio']:>10.3f}"
        )
    print()