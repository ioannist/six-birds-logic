"""Entropy-production and channel-information accounting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emergent_logic.markov import power, stationary_distribution, validate_kernel
from emergent_logic.metrics import induced_macro_kernel


def entropy_production_rate(
    P: np.ndarray,
    pi: np.ndarray | None = None,
    tol: float = 1e-15,
) -> float:
    """
    Return stationary entropy production rate (nats/step) for a discrete-time Markov chain.

    Uses:
        sigma = sum_{i,j} a_ij log(a_ij / a_ji),  where a_ij = pi_i * P_ij.
    """
    validate_kernel(P)
    arr = np.asarray(P, dtype=float)
    n = arr.shape[0]

    if pi is None:
        pi_arr = stationary_distribution(arr)
    else:
        pi_arr = np.asarray(pi, dtype=float)
        if pi_arr.ndim != 1 or pi_arr.shape[0] != n:
            raise ValueError(f"pi must be 1D with length {n}.")
        if not np.all(np.isfinite(pi_arr)):
            raise ValueError("pi contains non-finite values.")
        if float(pi_arr.min()) < -tol:
            raise ValueError("pi must be nonnegative within tolerance.")
        pi_arr = np.where((pi_arr < 0.0) & (pi_arr >= -tol), 0.0, pi_arr)
        total = float(pi_arr.sum())
        if total <= tol:
            raise ValueError("pi must have positive total mass.")
        pi_arr = pi_arr / total

    flux = pi_arr[:, None] * arr
    sigma = 0.0
    for i in range(n):
        for j in range(n):
            a_ij = float(flux[i, j])
            if a_ij <= tol:
                continue
            a_ji = float(flux[j, i])
            if a_ji <= tol:
                return float(np.inf)
            sigma += a_ij * np.log(a_ij / a_ji)

    if -tol < sigma < 0.0:
        return 0.0
    return float(sigma)


def apparent_entropy_production_rate(
    P: np.ndarray,
    f: np.ndarray,
    tau: int = 1,
    weights: np.ndarray | None = None,
    tol: float = 1e-15,
) -> float:
    """
    Compute apparent Markov EPR of a tau-step coarse-grained approximation induced by lens ``f``.
    """
    validate_kernel(P)
    arr = np.asarray(P, dtype=float)
    n = arr.shape[0]

    if weights is None:
        weight_arr = stationary_distribution(arr)
    else:
        weight_arr = np.asarray(weights, dtype=float)
        if weight_arr.ndim != 1 or weight_arr.shape[0] != n:
            raise ValueError(f"weights must be 1D with length {n}.")
        if not np.all(np.isfinite(weight_arr)):
            raise ValueError("weights contains non-finite values.")
        if float(weight_arr.min()) < -tol:
            raise ValueError("weights must be nonnegative within tolerance.")
        weight_arr = np.where((weight_arr < 0.0) & (weight_arr >= -tol), 0.0, weight_arr)

    K = induced_macro_kernel(arr, f, tau=tau, weights=weight_arr)
    row_sums = K.sum(axis=1)
    keep = np.where(row_sums > tol)[0]
    if keep.size == 0:
        return 0.0

    K_reduced = K[np.ix_(keep, keep)]
    if not np.allclose(K_reduced.sum(axis=1), 1.0, atol=max(tol, 1e-12), rtol=1e-12):
        raise ValueError("Reduced coarse-grained kernel rows are not stochastic.")
    return entropy_production_rate(K_reduced, tol=tol)


def channel_from_kernel(
    P: np.ndarray,
    f_input: np.ndarray,
    f_output: np.ndarray,
    tau: int = 1,
) -> np.ndarray:
    """
    Construct tau-step channel p(y|u) from kernel P using current-state inputs and future outputs.
    """
    validate_kernel(P)
    arr = np.asarray(P, dtype=float)
    n = arr.shape[0]

    fin = np.asarray(f_input)
    fout = np.asarray(f_output)
    if fin.ndim != 1 or fin.shape[0] != n:
        raise ValueError(f"f_input must be 1D with length {n}.")
    if fout.ndim != 1 or fout.shape[0] != n:
        raise ValueError(f"f_output must be 1D with length {n}.")
    if not np.issubdtype(fin.dtype, np.integer):
        raise ValueError("f_input must have integer dtype.")
    if not np.issubdtype(fout.dtype, np.integer):
        raise ValueError("f_output must have integer dtype.")
    if np.any(fin < 0) or np.any(fout < 0):
        raise ValueError("f_input/f_output labels must be nonnegative.")

    n_inputs = int(fin.max()) + 1
    n_outputs = int(fout.max()) + 1
    P_tau = power(arr, tau)

    channel = np.zeros((n_inputs, n_outputs), dtype=float)
    for u in range(n_inputs):
        idx = np.where(fin == u)[0]
        if idx.size == 0:
            raise ValueError(f"Input label {u} has no supporting microstates.")
        mu = np.zeros(n, dtype=float)
        mu[idx] = 1.0 / float(idx.size)
        dist = mu @ P_tau
        channel[u, :] = np.bincount(fout.astype(int, copy=False), weights=dist, minlength=n_outputs)

    if not np.allclose(channel.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12):
        raise RuntimeError("Constructed channel rows are not stochastic.")
    return channel


@dataclass(frozen=True)
class ChannelInfo:
    H_in: float
    H_out: float
    H_out_given_in: float
    I_in_out: float
    entropy_drop: float
    unretained_input_info: float


def channel_information_measures(
    channel: np.ndarray,
    p_input: np.ndarray | None = None,
    tol: float = 1e-15,
) -> ChannelInfo:
    """Compute Shannon information measures (bits) for a discrete channel."""
    c = np.asarray(channel, dtype=float)
    if c.ndim != 2 or c.shape[0] < 1 or c.shape[1] < 1:
        raise ValueError("channel must be a 2D array with positive dimensions.")
    if not np.all(np.isfinite(c)):
        raise ValueError("channel contains non-finite values.")
    if float(c.min()) < -tol:
        raise ValueError("channel entries must be nonnegative within tolerance.")
    c = np.where((c < 0.0) & (c >= -tol), 0.0, c)
    if not np.allclose(c.sum(axis=1), 1.0, atol=max(tol, 1e-12), rtol=1e-12):
        raise ValueError("channel rows must sum to 1 within tolerance.")

    n_inputs = c.shape[0]
    if p_input is None:
        p_in = np.full(n_inputs, 1.0 / n_inputs, dtype=float)
    else:
        p_in = np.asarray(p_input, dtype=float)
        if p_in.ndim != 1 or p_in.shape[0] != n_inputs:
            raise ValueError(f"p_input must be 1D with length {n_inputs}.")
        if not np.all(np.isfinite(p_in)):
            raise ValueError("p_input contains non-finite values.")
        if float(p_in.min()) < -tol:
            raise ValueError("p_input must be nonnegative within tolerance.")
        p_in = np.where((p_in < 0.0) & (p_in >= -tol), 0.0, p_in)
        total = float(p_in.sum())
        if total <= tol:
            raise ValueError("p_input must have positive total mass.")
        p_in = p_in / total

    p_joint = p_in[:, None] * c
    p_out = p_joint.sum(axis=0)

    h_in = _entropy_bits(p_in)
    h_out = _entropy_bits(p_out)
    h_out_given_in = float(np.dot(p_in, np.array([_entropy_bits(row) for row in c], dtype=float)))
    i_in_out = h_out - h_out_given_in

    entropy_drop = h_in - h_out
    unretained_input_info = h_in - i_in_out
    return ChannelInfo(
        H_in=float(h_in),
        H_out=float(h_out),
        H_out_given_in=float(h_out_given_in),
        I_in_out=float(i_in_out),
        entropy_drop=float(entropy_drop),
        unretained_input_info=float(unretained_input_info),
    )


def _entropy_bits(p: np.ndarray) -> float:
    arr = np.asarray(p, dtype=float)
    mask = arr > 0.0
    if not np.any(mask):
        return 0.0
    return float(-np.sum(arr[mask] * np.log2(arr[mask])))
