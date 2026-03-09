"""Finite-state Markov kernel utilities."""

from __future__ import annotations

import numpy as np


def validate_kernel(P: np.ndarray, tol: float = 1e-12, raise_on_fail: bool = True) -> bool:
    """Validate that ``P`` is a row-stochastic finite Markov kernel."""
    arr = np.asarray(P, dtype=float)

    if arr.ndim != 2:
        return _fail("Kernel must be a 2D array.", raise_on_fail)
    n_rows, n_cols = arr.shape
    if n_rows != n_cols or n_rows < 1:
        return _fail(
            f"Kernel must be square with shape (n, n), n>=1; got {arr.shape}.",
            raise_on_fail,
        )
    if not np.all(np.isfinite(arr)):
        return _fail("Kernel contains non-finite entries (NaN/inf).", raise_on_fail)

    min_entry = float(np.min(arr))
    if min_entry < -tol:
        return _fail(
            f"Kernel has entry below -tol: min={min_entry:.6g}, tol={tol:.6g}.",
            raise_on_fail,
        )

    row_sums = arr.sum(axis=1)
    max_row_err = float(np.max(np.abs(row_sums - 1.0)))
    if max_row_err > tol:
        return _fail(
            f"Kernel row sums must equal 1 within tol={tol:.6g}; max error={max_row_err:.6g}.",
            raise_on_fail,
        )

    return True


def normalize_kernel(
    P: np.ndarray,
    tol: float = 1e-15,
    on_zero_row: str = "error",
) -> np.ndarray:
    """Return a normalized copy of ``P`` with rows summing to one."""
    if on_zero_row not in {"error", "uniform"}:
        raise ValueError("on_zero_row must be one of {'error', 'uniform'}.")

    arr = np.array(P, dtype=float, copy=True)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 1:
        raise ValueError(f"Kernel must be square with shape (n, n), n>=1; got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Kernel contains non-finite entries (NaN/inf).")

    tiny_negative_mask = (arr < 0.0) & (arr >= -tol)
    arr[tiny_negative_mask] = 0.0

    min_entry = float(np.min(arr))
    if min_entry < -tol:
        raise ValueError(f"Kernel has entry below -tol after clipping: {min_entry:.6g}.")

    row_sums = arr.sum(axis=1)
    zero_rows = row_sums <= tol
    if np.any(zero_rows):
        if on_zero_row == "error":
            idx = np.flatnonzero(zero_rows)
            raise ValueError(f"Found row(s) with sum <= tol={tol:.6g}: {idx.tolist()}.")
        n = arr.shape[0]
        arr[zero_rows] = 1.0 / n
        row_sums = arr.sum(axis=1)

    arr = arr / row_sums[:, None]
    return arr


def power(P: np.ndarray, tau: int) -> np.ndarray:
    """Compute matrix power ``P**tau`` using exponentiation by squaring."""
    validate_kernel(P)
    if not isinstance(tau, (int, np.integer)):
        raise ValueError("tau must be an integer >= 0.")
    if tau < 0:
        raise ValueError("tau must be >= 0.")

    arr = np.asarray(P, dtype=float)
    n = arr.shape[0]
    result = np.eye(n, dtype=float)
    base = arr.copy()
    exp = int(tau)

    while exp > 0:
        if exp & 1:
            result = result @ base
        base = base @ base
        exp >>= 1

    return result


def simulate(
    P: np.ndarray,
    x0: int,
    steps: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate a state trajectory of length ``steps + 1`` from initial state ``x0``."""
    validate_kernel(P)
    if not isinstance(x0, (int, np.integer)):
        raise ValueError("x0 must be an integer index.")
    if not isinstance(steps, (int, np.integer)):
        raise ValueError("steps must be an integer >= 0.")
    if steps < 0:
        raise ValueError("steps must be >= 0.")

    arr = np.asarray(P, dtype=float)
    n = arr.shape[0]
    x = int(x0)
    if x < 0 or x >= n:
        raise ValueError(f"x0 must be in [0, {n - 1}] but got {x0}.")

    generator = np.random.default_rng() if rng is None else rng
    traj = np.empty(int(steps) + 1, dtype=int)
    traj[0] = x

    for t in range(1, int(steps) + 1):
        x = int(generator.choice(n, p=arr[x]))
        traj[t] = x

    return traj


def stationary_distribution(
    P: np.ndarray,
    method: str = "power",
    tol: float = 1e-12,
    max_iter: int = 1_000_000,
) -> np.ndarray:
    """Compute a stationary distribution ``pi`` for a finite Markov kernel."""
    validate_kernel(P)
    if method != "power":
        raise ValueError("Unsupported method. Supported methods: {'power'}.")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1.")

    arr = np.asarray(P, dtype=float)
    n = arr.shape[0]
    pi = np.full(n, 1.0 / n, dtype=float)

    for _ in range(int(max_iter)):
        pi_next = pi @ arr
        if np.linalg.norm(pi_next - pi, ord=1) <= tol:
            pi_next = np.clip(pi_next, 0.0, None)
            total = float(pi_next.sum())
            if total <= 0.0:
                raise RuntimeError("Degenerate stationary distribution encountered.")
            return pi_next / total
        pi = pi_next

    raise RuntimeError(
        f"stationary_distribution did not converge within {max_iter} iterations "
        f"(tol={tol})."
    )


def _fail(message: str, raise_on_fail: bool) -> bool:
    if raise_on_fail:
        raise ValueError(message)
    return False
