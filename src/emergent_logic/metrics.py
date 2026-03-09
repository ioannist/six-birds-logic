"""Closure and route-mismatch diagnostics for coarse-grained Markov systems."""

from __future__ import annotations

import numpy as np

from emergent_logic.endomap import U_f
from emergent_logic.lens import num_macro, pushforward, validate_lens
from emergent_logic.markov import power, validate_kernel


def micro_to_macro_rows(
    P: np.ndarray,
    f: np.ndarray,
    tau: int = 1,
    n_macro: int | None = None,
) -> np.ndarray:
    """Return per-microstate macro transition rows after ``tau`` steps."""
    validate_kernel(P)
    labels = np.asarray(f)
    if n_macro is None:
        validate_lens(labels)
        macro_count = num_macro(labels)
    else:
        validate_lens(labels, n_macro=n_macro)
        macro_count = int(n_macro)

    P_tau = power(P, tau)
    n_micro = P_tau.shape[0]
    if labels.shape[0] != n_micro:
        raise ValueError(f"lens length ({labels.shape[0]}) must equal P size ({n_micro}).")

    indicator = np.zeros((n_micro, macro_count), dtype=float)
    indicator[np.arange(n_micro, dtype=int), labels] = 1.0
    rows = P_tau @ indicator

    if not np.allclose(rows.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12):
        raise RuntimeError("micro_to_macro_rows are not row-stochastic.")
    return rows


def induced_macro_kernel(
    P: np.ndarray,
    f: np.ndarray,
    tau: int = 1,
    n_macro: int | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate induced macro kernel by averaging micro macro-rows inside each fiber."""
    labels = np.asarray(f)
    if n_macro is None:
        validate_lens(labels)
        macro_count = num_macro(labels)
    else:
        validate_lens(labels, n_macro=n_macro)
        macro_count = int(n_macro)

    rows = micro_to_macro_rows(P, labels, tau=tau, n_macro=macro_count)
    n_micro = rows.shape[0]

    weight_arr: np.ndarray | None = None
    if weights is not None:
        weight_arr = np.asarray(weights, dtype=float)
        if weight_arr.ndim != 1:
            raise ValueError("weights must be a 1D array when provided.")
        if weight_arr.shape[0] != n_micro:
            raise ValueError(f"weights length ({weight_arr.shape[0]}) must equal n_micro ({n_micro}).")
        if not np.all(np.isfinite(weight_arr)):
            raise ValueError("weights contains non-finite entries.")
        if float(weight_arr.min()) < 0.0:
            raise ValueError("weights must be nonnegative.")

    kernel = np.zeros((macro_count, macro_count), dtype=float)
    for x in range(macro_count):
        idx = np.where(labels == x)[0]
        if idx.size == 0:
            continue
        if weight_arr is None:
            kernel[x, :] = rows[idx, :].mean(axis=0)
            continue

        local_w = weight_arr[idx]
        w_sum = float(local_w.sum())
        if w_sum <= 0.0:
            raise ValueError(f"weights on nonempty fiber {x} must sum to > 0.")
        kernel[x, :] = (local_w[:, None] * rows[idx, :]).sum(axis=0) / w_sum

    nonempty_rows = np.any(kernel > 0.0, axis=1)
    if np.any(nonempty_rows):
        row_sums = kernel[nonempty_rows].sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-12, rtol=1e-12):
            raise RuntimeError("induced macro kernel has nonempty rows that are not stochastic.")
    return kernel


def route_mismatch(
    P: np.ndarray,
    f: np.ndarray,
    tau: int = 1,
    n_macro: int | None = None,
    weights: np.ndarray | None = None,
    return_per_macro: bool = False,
) -> float | tuple[float, np.ndarray]:
    """Compute route mismatch across fibers, with optional per-macro values."""
    labels = np.asarray(f)
    if n_macro is None:
        validate_lens(labels)
        macro_count = num_macro(labels)
    else:
        validate_lens(labels, n_macro=n_macro)
        macro_count = int(n_macro)

    rows = micro_to_macro_rows(P, labels, tau=tau, n_macro=macro_count)
    kernel = induced_macro_kernel(P, labels, tau=tau, n_macro=macro_count, weights=weights)
    n_micro = rows.shape[0]

    weight_arr: np.ndarray | None = None
    if weights is not None:
        weight_arr = np.asarray(weights, dtype=float)
        if weight_arr.ndim != 1:
            raise ValueError("weights must be a 1D array when provided.")
        if weight_arr.shape[0] != n_micro:
            raise ValueError(f"weights length ({weight_arr.shape[0]}) must equal n_micro ({n_micro}).")
        if not np.all(np.isfinite(weight_arr)):
            raise ValueError("weights contains non-finite entries.")
        if float(weight_arr.min()) < 0.0:
            raise ValueError("weights must be nonnegative.")

    per_macro = np.zeros(macro_count, dtype=float)
    macro_mass = np.zeros(macro_count, dtype=float)

    for x in range(macro_count):
        idx = np.where(labels == x)[0]
        if idx.size == 0:
            continue
        d = np.sum(np.abs(rows[idx, :] - kernel[x, :]), axis=1)
        if weight_arr is None:
            per_macro[x] = float(d.mean())
            macro_mass[x] = float(idx.size)
        else:
            local_w = weight_arr[idx]
            w_sum = float(local_w.sum())
            if w_sum <= 0.0:
                raise ValueError(f"weights on nonempty fiber {x} must sum to > 0.")
            per_macro[x] = float(np.dot(local_w, d) / w_sum)
            macro_mass[x] = w_sum

    denom = float(macro_mass.sum())
    if denom <= 0.0:
        raise ValueError("total macro mass must be > 0.")
    global_rm = float(np.dot(macro_mass, per_macro) / denom)

    if return_per_macro:
        return global_rm, per_macro
    return global_rm


def distribution_commutation_defect(
    mu: np.ndarray,
    P: np.ndarray,
    f: np.ndarray,
    tau: int = 1,
    prototypes: object | None = None,
    n_macro: int | None = None,
) -> float:
    """Return ``||Q_f(mu P^tau) - Q_f(U_f(Q_f(mu)) P^tau)||_1``."""
    validate_kernel(P)
    labels = np.asarray(f)
    if n_macro is None:
        validate_lens(labels)
        macro_count = num_macro(labels)
    else:
        validate_lens(labels, n_macro=n_macro)
        macro_count = int(n_macro)

    mu_arr = np.asarray(mu, dtype=float)
    if mu_arr.ndim != 1:
        raise ValueError("mu must be a 1D array.")
    if mu_arr.shape[0] != labels.shape[0]:
        raise ValueError(
            f"mu length ({mu_arr.shape[0]}) must equal lens length ({labels.shape[0]})."
        )
    if not np.all(np.isfinite(mu_arr)):
        raise ValueError("mu contains non-finite entries.")

    P_tau = power(P, tau)
    nu1 = pushforward(mu_arr @ P_tau, labels, n_macro=macro_count)

    nu0 = pushforward(mu_arr, labels, n_macro=macro_count)
    mu0 = U_f(nu0, labels, prototypes=prototypes, n_macro=macro_count)
    nu2 = pushforward(mu0 @ P_tau, labels, n_macro=macro_count)

    return float(np.sum(np.abs(nu1 - nu2)))
