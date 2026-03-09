"""Predicate stability and binary gate inference utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emergent_logic.markov import power, validate_kernel


def bits_to_index(bits: np.ndarray) -> np.ndarray:
    """
    Convert bits in {0,1} to integer indices using MSB-first convention.

    Accepted shapes are (k,) and (n, k). Returns shape () for (k,) input
    and shape (n,) for (n, k) input.
    """
    arr = np.asarray(bits)
    if arr.ndim not in (1, 2):
        raise ValueError("bits must have shape (k,) or (n, k).")
    if arr.size == 0:
        raise ValueError("bits must be non-empty.")
    if not np.issubdtype(arr.dtype, np.integer) and arr.dtype != np.bool_:
        raise ValueError("bits must have integer or bool dtype.")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("bits entries must be in {0, 1}.")

    if arr.ndim == 1:
        k = arr.shape[0]
        powers = (1 << np.arange(k - 1, -1, -1, dtype=int))
        return np.array(int(np.dot(arr.astype(int), powers)), dtype=int)

    k = arr.shape[1]
    powers = (1 << np.arange(k - 1, -1, -1, dtype=int))
    return (arr.astype(int) * powers).sum(axis=1, dtype=int)


def index_to_bits(index: int | np.ndarray, k: int) -> np.ndarray:
    """
    Convert integer indices in [0, 2**k) to bits using MSB-first convention.

    Returns shape (k,) for scalar index, or shape (n, k) for vector input.
    """
    if not isinstance(k, (int, np.integer)) or int(k) < 1:
        raise ValueError("k must be an integer >= 1.")
    k_int = int(k)
    max_idx = 1 << k_int

    idx_arr = np.asarray(index)
    if idx_arr.ndim > 1:
        raise ValueError("index must be a scalar or a 1D array.")
    if not np.issubdtype(idx_arr.dtype, np.integer):
        raise ValueError("index must have integer dtype.")
    if idx_arr.size == 0:
        raise ValueError("index must be non-empty.")
    if np.any(idx_arr < 0) or np.any(idx_arr >= max_idx):
        raise ValueError(f"index values must lie in [0, {max_idx}).")

    shifts = np.arange(k_int - 1, -1, -1, dtype=int)
    bits = ((idx_arr[..., None].astype(int) >> shifts) & 1).astype(int)
    if np.isscalar(index):
        return bits.reshape(k_int)
    return bits


def predicate_stability_kernel(
    P: np.ndarray,
    pred: np.ndarray,
    tau: int = 1,
    mu: np.ndarray | None = None,
    tol: float = 1e-15,
) -> float:
    """
    Compute Pr[b_{t+tau} = b_t] for predicate values under kernel evolution.

    If ``mu`` is None, uniform distribution over microstates is used.
    """
    validate_kernel(P)
    P_tau = power(P, tau)
    n_micro = P_tau.shape[0]

    pred_arr = np.asarray(pred)
    if pred_arr.ndim != 1 or pred_arr.shape[0] != n_micro:
        raise ValueError(f"pred must be 1D with length {n_micro}.")
    if not np.all((pred_arr == 0) | (pred_arr == 1)):
        raise ValueError("pred values must be in {0, 1} or bool.")

    if mu is None:
        mu_arr = np.full(n_micro, 1.0 / n_micro, dtype=float)
    else:
        mu_arr = np.asarray(mu, dtype=float)
        if mu_arr.ndim != 1 or mu_arr.shape[0] != n_micro:
            raise ValueError(f"mu must be 1D with length {n_micro}.")
        if not np.all(np.isfinite(mu_arr)):
            raise ValueError("mu contains non-finite values.")
        if float(mu_arr.min()) < -tol:
            raise ValueError("mu must be nonnegative within tolerance.")
        mu_arr = np.where((mu_arr < 0.0) & (mu_arr >= -tol), 0.0, mu_arr)
        mu_sum = float(mu_arr.sum())
        if mu_sum <= tol:
            raise ValueError("mu must have positive total mass.")
        mu_arr = mu_arr / mu_sum

    eq = (pred_arr[:, None] == pred_arr[None, :]).astype(float)
    return float((mu_arr[:, None] * P_tau * eq).sum())


def predicate_stability_trajectory(pred_values: np.ndarray, tau: int = 1) -> float:
    """
    Empirical Pr[b_{t+tau} = b_t] from a 1D predicate trajectory.

    Returns NaN when the trajectory is not long enough for the given ``tau``.
    """
    if not isinstance(tau, (int, np.integer)) or int(tau) < 0:
        raise ValueError("tau must be an integer >= 0.")
    tau_int = int(tau)

    arr = np.asarray(pred_values)
    if arr.ndim != 1:
        raise ValueError("pred_values must be a 1D array.")
    if tau_int == 0:
        if arr.shape[0] == 0:
            return float("nan")
        return 1.0
    if arr.shape[0] <= tau_int:
        return float("nan")
    return float(np.mean(arr[:-tau_int] == arr[tau_int:]))


@dataclass(frozen=True)
class GateFit:
    """Summary of a fitted binary-output gate."""

    k: int
    table: np.ndarray
    confusion: np.ndarray
    error_rate: float
    H_out_given_in: float
    I_in_out: float


def fit_gate_from_samples(
    inputs: np.ndarray,
    outputs: np.ndarray,
    k: int | None = None,
    smoothing: float = 0.0,
) -> GateFit:
    """
    Fit a k-input -> 1-output gate from sample pairs.

    ``inputs`` can be bit patterns of shape (n, k) or integer indices of shape (n,).
    ``outputs`` must be shape (n,) with values in {0,1}.
    """
    if smoothing < 0.0:
        raise ValueError("smoothing must be >= 0.")

    out = np.asarray(outputs)
    if out.ndim != 1:
        raise ValueError("outputs must be a 1D array.")
    n = out.shape[0]
    if n < 1:
        raise ValueError("at least one sample is required.")
    if not np.issubdtype(out.dtype, np.integer) and out.dtype != np.bool_:
        raise ValueError("outputs must have integer or bool dtype.")
    if not np.all((out == 0) | (out == 1)):
        raise ValueError("outputs must be in {0,1}.")
    out_int = out.astype(int, copy=False)

    inp = np.asarray(inputs)
    if inp.ndim == 2:
        inferred_k = inp.shape[1]
        if inferred_k < 1:
            raise ValueError("inputs bit matrix must have at least one column.")
        if inp.shape[0] != n:
            raise ValueError("inputs and outputs must have matching first dimension.")
        if not np.issubdtype(inp.dtype, np.integer) and inp.dtype != np.bool_:
            raise ValueError("inputs bits must have integer or bool dtype.")
        if not np.all((inp == 0) | (inp == 1)):
            raise ValueError("inputs bits must be in {0,1}.")
        if k is None:
            k_int = int(inferred_k)
        else:
            if not isinstance(k, (int, np.integer)) or int(k) < 1:
                raise ValueError("k must be an integer >= 1.")
            k_int = int(k)
            if k_int != inferred_k:
                raise ValueError("provided k does not match inputs.shape[1].")
        idx = bits_to_index(inp).astype(int, copy=False)
    elif inp.ndim == 1:
        if inp.shape[0] != n:
            raise ValueError("inputs and outputs must have matching length.")
        if k is None:
            raise ValueError("k must be provided when inputs are integer indices.")
        if not isinstance(k, (int, np.integer)) or int(k) < 1:
            raise ValueError("k must be an integer >= 1.")
        k_int = int(k)
        if not np.issubdtype(inp.dtype, np.integer):
            raise ValueError("index-form inputs must have integer dtype.")
        idx = inp.astype(int, copy=False)
        if np.any(idx < 0) or np.any(idx >= (1 << k_int)):
            raise ValueError(f"index-form inputs must lie in [0, {1 << k_int}).")
    else:
        raise ValueError("inputs must have shape (n,k) bits or shape (n,) indices.")

    n_rows = 1 << k_int
    confusion = np.zeros((n_rows, 2), dtype=float)
    np.add.at(confusion, (idx, out_int), 1.0)

    table = (confusion[:, 1] > confusion[:, 0]).astype(int)

    row_sums = confusion.sum(axis=1)
    total = float(n)
    errors = float((row_sums - np.max(confusion, axis=1)).sum())
    error_rate = errors / total

    confusion_eff = confusion + float(smoothing) if smoothing > 0.0 else confusion.copy()
    row_sums_eff = confusion_eff.sum(axis=1)
    total_eff = float(row_sums_eff.sum())

    p_row = confusion_eff / row_sums_eff[:, None]
    H_rows = _binary_entropy(p_row[:, 0], p_row[:, 1])
    H_out_given_in = float(np.dot(row_sums_eff / total_eff, H_rows))

    p_out = confusion.sum(axis=0) / total
    H_out = float(_binary_entropy(np.array([p_out[0]]), np.array([p_out[1]]))[0])
    I_in_out = H_out - H_out_given_in

    return GateFit(
        k=k_int,
        table=table,
        confusion=confusion.astype(int),
        error_rate=float(error_rate),
        H_out_given_in=float(H_out_given_in),
        I_in_out=float(I_in_out),
    )


def _binary_entropy(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Binary entropy in bits with 0*log2(0) treated as 0."""
    out = np.zeros_like(p0, dtype=float)
    mask0 = p0 > 0.0
    mask1 = p1 > 0.0
    out[mask0] -= p0[mask0] * np.log2(p0[mask0])
    out[mask1] -= p1[mask1] * np.log2(p1[mask1])
    return out
