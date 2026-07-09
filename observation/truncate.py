"""
observation/truncate.py
-----------------------
The "drop energy" intervention.

To relate weight energy to task metrics we deliberately remove spectral energy from
every conv/linear weight and measure what happens. Removal is done by Eckart-Young
low-rank truncation: keep the top-k singular values that retain a fraction `tau` of
each matrix's Frobenius energy, discard the rest, and reconstruct.

  W  = U S Vᵀ                         (full SVD)
  keep smallest k s.t.  Σ_{i<=k} σ_i²  >=  tau · Σ σ_i²
  W' = U[:, :k] diag(S[:k]) V[:, :k]ᵀ

tau = 1.0 leaves the matrix untouched (baseline). Lower tau drops more energy.
Conv weights [out, in, kh, kw] are flattened to [out, in*kh*kw] for the SVD — the
same 2-D view the Phase-1 audit measured energy on — then reshaped back.
"""

import copy
from typing import Dict, Tuple

import torch
from torch import nn


def truncate_matrix(W: torch.Tensor, tau: float) -> Tuple[torch.Tensor, float, int]:
    """
    Low-rank truncate a 2-D matrix to retain `tau` of its Frobenius energy.

    Returns:
        W_approx        : rank-k reconstruction, same shape/dtype/device as W
        retained_energy : fraction of energy actually kept (>= tau, exactly 1.0 if tau>=1)
        k               : number of singular values kept
    """
    if tau >= 1.0:
        return W.clone(), 1.0, min(W.shape)

    W32 = W.detach().to(torch.float32)
    U, S, Vt = torch.linalg.svd(W32, full_matrices=False)
    total = (S ** 2).sum()
    cum = torch.cumsum(S ** 2, dim=0) / total
    # First index whose cumulative energy reaches tau (+1 for count).
    k = int(torch.searchsorted(cum, torch.tensor(tau)).item()) + 1
    k = max(1, min(k, S.shape[0]))

    W_approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
    retained = float(cum[k - 1].item())
    return W_approx.to(W.dtype).to(W.device), retained, k


def truncate_model(
    model: nn.Module,
    tau: float,
    mode: str = "conv_linear",
) -> Tuple[nn.Module, Dict[str, dict]]:
    """
    Return a deep copy of `model` with every eligible weight low-rank truncated at `tau`.

    Args:
        model : source model (left unmodified)
        tau   : energy fraction to retain per layer (1.0 = identity copy)
        mode  : "conv_linear" (only nn.Conv2d/nn.Linear) or "all" (any .weight)

    Returns:
        model_trunc : the truncated deep copy
        info        : per-layer {retained_energy, k, shape} for auditing the sweep
    """
    model_trunc = copy.deepcopy(model)
    info: Dict[str, dict] = {}  # per-layer audit of the truncation

    for name, module in model_trunc.named_modules():
        if mode == "conv_linear" and not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue

        # isinstance narrows `weight` to Tensor (a Parameter is a Tensor subclass),
        # which also skips modules that carry no weight in "all" mode.
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue

        W = weight.data
        orig_shape = tuple(W.shape)
        W2d = W.flatten(1) if W.dim() == 4 else W
        if W2d.dim() < 2:
            continue

        W_approx, retained, k = truncate_matrix(W2d, tau)
        weight.data = W_approx.reshape(orig_shape)
        info[name] = {"retained_energy": retained, "k": k, "shape": orig_shape}

    return model_trunc, info


def global_retained_energy(info: Dict[str, dict]) -> float:
    """
    Aggregate per-layer retention into one number: mean retained energy across the
    truncated layers. Reported on the x-axis of the energy-vs-metric curves.
    """
    if not info:
        return 1.0
    vals = [v["retained_energy"] for v in info.values()]
    return float(sum(vals) / len(vals))
