"""Unsupervised binary partition discovery from Markov transition structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emergent_logic.gates import predicate_stability_kernel
from emergent_logic.markov import power, stationary_distribution, validate_kernel
from emergent_logic.metrics import route_mismatch


@dataclass(frozen=True)
class BinaryPartitionCandidate:
    labels: np.ndarray
    threshold: float
    eigenvalue2: float
    metastability: float
    rm: float
    score: float
    size0: int
    size1: int


def partition_agreement(labels: np.ndarray, truth: np.ndarray) -> float:
    """Return binary partition agreement up to label swap."""
    a = _as_binary_labels(labels, "labels")
    b = _as_binary_labels(truth, "truth")
    if a.shape != b.shape:
        raise ValueError("labels and truth must have the same shape.")
    same = float(np.mean(a == b))
    flipped = float(np.mean((1 - a) == b))
    return max(same, flipped)


def spectral_second_vector(
    P: np.ndarray,
    tau: int = 1,
    pi: np.ndarray | None = None,
    tol: float = 1e-15,
) -> tuple[float, np.ndarray]:
    """Compute second eigenpair of reversibilized/symmetrized tau-step operator."""
    validate_kernel(P)
    arr = np.asarray(P, dtype=float)
    n = arr.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 states to compute a second spectral vector.")

    P_tau = power(arr, tau)
    pi_arr = stationary_distribution(arr) if pi is None else _normalize_prob(pi, n, tol)

    sqrt_pi = np.sqrt(np.maximum(pi_arr, tol))
    inv_sqrt = 1.0 / sqrt_pi

    M = (sqrt_pi[:, None] * P_tau) * inv_sqrt[None, :]
    S = 0.5 * (M + M.T)

    eigvals, eigvecs = np.linalg.eigh(S)
    order = np.argsort(eigvals)[::-1]
    eig2 = float(eigvals[order[1]])
    v2 = eigvecs[:, order[1]].astype(float, copy=True)

    # Fix sign deterministically; eigensolvers may otherwise flip signs across runs.
    i_ref = int(np.argmax(np.abs(v2)))
    if v2[i_ref] < 0.0:
        v2 = -v2

    return eig2, v2


def binary_thresholds_from_vector(
    v: np.ndarray,
    max_thresholds: int | None = None,
    tol: float = 1e-12,
) -> np.ndarray:
    """Generate deterministic threshold candidates from midpoints between unique values."""
    arr = np.asarray(v, dtype=float)
    if arr.ndim != 1:
        raise ValueError("v must be a 1D array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("v contains non-finite values.")

    uniq = np.unique(arr)
    if uniq.size < 2:
        return np.array([], dtype=float)

    mids = 0.5 * (uniq[:-1] + uniq[1:])
    if mids.size == 0:
        return np.array([], dtype=float)

    kept = [float(mids[0])]
    for x in mids[1:]:
        if abs(float(x) - kept[-1]) > tol:
            kept.append(float(x))
    thresholds = np.array(kept, dtype=float)

    if max_thresholds is not None:
        if not isinstance(max_thresholds, (int, np.integer)) or int(max_thresholds) < 1:
            raise ValueError("max_thresholds must be an integer >= 1 when provided.")
        m = int(max_thresholds)
        if thresholds.size > m:
            idx = np.linspace(0, thresholds.size - 1, num=m)
            idx = np.unique(np.rint(idx).astype(int))
            thresholds = thresholds[idx]

    return np.sort(thresholds)


def spectral_binary_candidates(
    P: np.ndarray,
    tau: int = 1,
    max_thresholds: int | None = None,
    pi: np.ndarray | None = None,
    tol: float = 1e-15,
) -> list[BinaryPartitionCandidate]:
    """Generate/scored binary partitions by thresholding the second spectral vector."""
    validate_kernel(P)
    arr = np.asarray(P, dtype=float)
    n = arr.shape[0]

    pi_arr = stationary_distribution(arr) if pi is None else _normalize_prob(pi, n, tol)
    eigenvalue2, vec2 = spectral_second_vector(arr, tau=tau, pi=pi_arr, tol=tol)
    thresholds = binary_thresholds_from_vector(vec2, max_thresholds=max_thresholds)

    candidates: list[BinaryPartitionCandidate] = []
    seen: set[bytes] = set()

    for threshold in thresholds:
        labels = (vec2 > threshold).astype(int)
        if np.all(labels == 0) or np.all(labels == 1):
            continue

        if labels[0] == 1:
            labels = 1 - labels

        key = labels.tobytes()
        if key in seen:
            continue
        seen.add(key)

        metastability = float(predicate_stability_kernel(arr, labels, tau=tau, mu=pi_arr))
        rm = float(route_mismatch(arr, labels, tau=tau, weights=pi_arr))
        score = float(metastability - rm)
        size0 = int(np.sum(labels == 0))
        size1 = int(np.sum(labels == 1))

        candidates.append(
            BinaryPartitionCandidate(
                labels=labels.astype(int, copy=True),
                threshold=float(threshold),
                eigenvalue2=float(eigenvalue2),
                metastability=metastability,
                rm=rm,
                score=score,
                size0=size0,
                size1=size1,
            )
        )

    candidates.sort(
        key=lambda c: (
            -float(c.score),
            -float(c.metastability),
            float(c.rm),
            abs(float(c.threshold)),
        )
    )
    return candidates


def best_binary_partition(
    P: np.ndarray,
    tau: int = 1,
    max_thresholds: int | None = None,
    pi: np.ndarray | None = None,
    tol: float = 1e-15,
) -> BinaryPartitionCandidate:
    """Return highest-scoring discovered binary partition."""
    candidates = spectral_binary_candidates(
        P,
        tau=tau,
        max_thresholds=max_thresholds,
        pi=pi,
        tol=tol,
    )
    if not candidates:
        raise RuntimeError("No nontrivial binary partition candidates were generated.")
    return candidates[0]


def _as_binary_labels(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if arr.size < 1:
        raise ValueError(f"{name} must be non-empty.")
    if not np.issubdtype(arr.dtype, np.integer) and arr.dtype != np.bool_:
        raise ValueError(f"{name} must have integer/bool dtype.")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError(f"{name} values must be binary (0/1).")
    return arr.astype(int, copy=False)


def _normalize_prob(pi: np.ndarray, n: int, tol: float) -> np.ndarray:
    arr = np.asarray(pi, dtype=float)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"pi must be 1D with length {n}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("pi contains non-finite values.")
    if float(arr.min()) < -tol:
        raise ValueError("pi must be nonnegative within tolerance.")
    arr = np.where((arr < 0.0) & (arr >= -tol), 0.0, arr)
    s = float(arr.sum())
    if s <= tol:
        raise ValueError("pi must have positive total mass.")
    return arr / s
