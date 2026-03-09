"""Coarse-graining lens and fiber utilities."""

from __future__ import annotations

import numpy as np


def validate_lens(
    f: np.ndarray,
    n_macro: int | None = None,
    raise_on_fail: bool = True,
) -> bool:
    """Validate a lens mapping from microstate index to nonnegative macro label."""
    arr = np.asarray(f)

    if arr.ndim != 1:
        return _fail("Lens must be a 1D array.", raise_on_fail)
    if arr.size < 1:
        return _fail("Lens must contain at least one microstate.", raise_on_fail)
    if not np.issubdtype(arr.dtype, np.integer):
        return _fail("Lens entries must have integer dtype.", raise_on_fail)
    if not np.all(np.isfinite(arr)):
        return _fail("Lens contains non-finite entries.", raise_on_fail)

    if int(arr.min()) < 0:
        return _fail("Lens labels must be >= 0.", raise_on_fail)

    if n_macro is not None:
        if not isinstance(n_macro, (int, np.integer)):
            return _fail("n_macro must be an integer >= 1.", raise_on_fail)
        if int(n_macro) < 1:
            return _fail("n_macro must be >= 1.", raise_on_fail)
        if int(arr.max()) >= int(n_macro):
            return _fail(
                f"Lens label max={int(arr.max())} must be < n_macro={int(n_macro)}.",
                raise_on_fail,
            )

    return True


def num_macro(f: np.ndarray) -> int:
    """Return the implied number of macro labels as ``max(f) + 1``."""
    validate_lens(f)
    arr = np.asarray(f)
    return int(arr.max()) + 1


def fibers(f: np.ndarray, n_macro: int | None = None) -> list[np.ndarray]:
    """Return microstate index fibers for each macro label."""
    arr = np.asarray(f)
    if n_macro is None:
        validate_lens(arr)
        macro_count = num_macro(arr)
    else:
        validate_lens(arr, n_macro=n_macro)
        macro_count = int(n_macro)

    return [np.where(arr == label)[0].astype(int, copy=False) for label in range(macro_count)]


def pushforward(mu: np.ndarray, f: np.ndarray, n_macro: int | None = None) -> np.ndarray:
    """Push forward a micro-distribution ``mu`` along lens ``f`` to macro space."""
    labels = np.asarray(f)
    if n_macro is None:
        validate_lens(labels)
        macro_count = num_macro(labels)
    else:
        validate_lens(labels, n_macro=n_macro)
        macro_count = int(n_macro)

    weights = np.asarray(mu, dtype=float)
    if weights.ndim != 1:
        raise ValueError("mu must be a 1D array.")
    if weights.shape[0] != labels.shape[0]:
        raise ValueError(
            f"mu length ({weights.shape[0]}) must equal lens length ({labels.shape[0]})."
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("mu contains non-finite entries.")

    out = np.bincount(labels, weights=weights, minlength=macro_count).astype(float, copy=False)
    if not np.isclose(out.sum(), weights.sum(), atol=1e-12, rtol=1e-12):
        raise RuntimeError("Mass was not preserved in pushforward.")
    return out


def is_definable_predicate(pred: np.ndarray, f: np.ndarray) -> bool:
    """Return whether ``pred`` is constant on every fiber of ``f``."""
    labels = np.asarray(f)
    validate_lens(labels)

    values = np.asarray(pred)
    if values.ndim != 1:
        raise ValueError("pred must be a 1D array.")
    if values.shape[0] != labels.shape[0]:
        raise ValueError(
            f"pred length ({values.shape[0]}) must equal lens length ({labels.shape[0]})."
        )

    for idx in fibers(labels):
        if idx.size == 0:
            continue
        fiber_vals = values[idx]
        if not np.all(fiber_vals == fiber_vals[0]):
            return False
    return True


def _fail(message: str, raise_on_fail: bool) -> bool:
    if raise_on_fail:
        raise ValueError(message)
    return False
