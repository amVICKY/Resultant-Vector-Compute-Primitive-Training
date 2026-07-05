"""
core/subspace.py
----------------
Phase 2 tools: significant subspace extraction, cross-layer principal angles,
and resultant computation in the jointly significant subspace.

These are STUBS, populated after Phase 1 confirms the need for the subspace
approach (i.e., rank-1 energy and alignment are consistently low).

Mathematical objects used:
  - Truncated SVD for significant basis (Eckart-Young theorem)
  - Principal angles via SVD of the cross-subspace alignment matrix
  - Resultant as alignment-weighted sum within the jointly significant subspace
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


def get_significant_basis(
    W: torch.Tensor,
    tau: float = 0.90
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compute the significant subspace of a weight matrix at energy threshold τ.

    Args:
        W   : weight matrix [m, n]
        tau : energy threshold (default 0.90 = 90% energy retained)

    Returns:
        U_k : output basis  [m, k]   — directions this layer outputs significantly
        S_k : singular values [k]
        V_k : input basis   [n, k]   — directions this layer responds to significantly
        k   : significant rank

    Usage:
        U_k, S_k, V_k, k = get_significant_basis(W, tau=0.90)
        W_approx = U_k @ np.diag(S_k) @ V_k.T   # rank-k approximation
    """
    W_np = W.numpy() if isinstance(W, torch.Tensor) else W
    U, S, Vt = np.linalg.svd(W_np, full_matrices=False)

    # Find minimum k for τ energy
    energy = (S ** 2).cumsum() / (S ** 2).sum()
    k = int(np.searchsorted(energy, tau)) + 1
    k = min(k, len(S))

    return U[:, :k], S[:k], Vt[:k, :].T, k   # V_k columns = right singular vectors


def cross_layer_principal_angles(
    U_l: np.ndarray,
    V_lp1: np.ndarray,
    gamma: float = 0.70
) -> Dict:
    """
    Compute principal angles between output subspace of layer l
    and input subspace of layer l+1.

    Args:
        U_l   : output basis of layer l     [m, k_l]
        V_lp1 : input basis of layer l+1    [m, k_{l+1}]
        gamma : alignment threshold for "jointly significant" (default 0.7)

    Returns dict:
        cos_angles      : cosines of all principal angles, sorted descending
        angles_deg      : principal angles in degrees
        r               : number of jointly significant directions (cos ≥ gamma)
        P               : aligned output directions  [k_l, r]
        Q               : aligned input directions   [k_{l+1}, r]
        A               : raw alignment matrix [k_l, k_{l+1}]

    Math:
        A = U_lᵀ V_{l+1}         [k_l × k_{l+1}]
        A = P Λ Qᵀ  (SVD)
        cos θᵢ = λᵢ  (diagonal of Λ)
    """
    # NOTE: This is a stub — implement once Phase 1 confirms low rank-1 energy.
    # The alignment matrix A lives in a k_l × k_{l+1} space (small SVD).
    raise NotImplementedError(
        "Phase 2 not yet active. Run Phase 1 (exp1_rank1_audit.py) first.\n"
        "Once rank-1 energy is confirmed low, implement this function."
    )


def compute_resultant(
    U_l: np.ndarray,
    V_lp1: np.ndarray,
    gamma: float = 0.70
) -> Tuple[Optional[np.ndarray], int]:
    """
    Compute the resultant direction in the jointly significant subspace
    between layer l and layer l+1.

    The resultant is the alignment-weighted sum of jointly significant directions:
        g_l = Σᵢ (cos θᵢ) · ũᵢ
    where ũᵢ are the aligned output directions (columns of U_l @ P).

    Returns:
        g   : normalized resultant vector [m], or None if r=0
        r   : number of jointly significant directions used

    This is a stub — implement after Phase 2 principal angle analysis confirms
    that jointly significant subspaces exist (r_l > 0 for most layer pairs).
    """
    raise NotImplementedError(
        "Phase 2 not yet active. Implement after Phase 2 experiments confirm "
        "cross-layer subspace coherence."
    )