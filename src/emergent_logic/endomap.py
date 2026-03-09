"""Canonical lift and empirical endomap utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from emergent_logic.lens import fibers, num_macro, pushforward, validate_lens
from emergent_logic.markov import power, validate_kernel


def uniform_prototypes(
    f: np.ndarray,
    n_macro: int | None = None,
    tol: float = 1e-15,
) -> list[np.ndarray]:
    """Construct default uniform fiber prototypes in global micro coordinates."""
    labels = np.asarray(f)
    if n_macro is None:
        validate_lens(labels)
        macro_count = num_macro(labels)
    else:
        validate_lens(labels, n_macro=n_macro)
        macro_count = int(n_macro)

    n_micro = int(labels.shape[0])
    grouped = fibers(labels, n_macro=macro_count)
    out: list[np.ndarray] = []
    for idx in grouped:
        proto = np.zeros(n_micro, dtype=float)
        if idx.size > 0:
            proto[idx] = 1.0 / float(idx.size)
        out.append(proto)

    _validate_prototypes(out, grouped, n_micro, tol)
    return out


def U_f(
    nu: np.ndarray,
    f: np.ndarray,
    prototypes: object | None = None,
    n_macro: int | None = None,
    tol: float = 1e-15,
) -> np.ndarray:
    """Lift macro distribution ``nu`` onto micro space using per-fiber prototypes."""
    labels = np.asarray(f)
    if n_macro is None:
        validate_lens(labels)
        macro_count = num_macro(labels)
    else:
        validate_lens(labels, n_macro=n_macro)
        macro_count = int(n_macro)

    grouped = fibers(labels, n_macro=macro_count)
    n_micro = int(labels.shape[0])

    nu_arr = np.asarray(nu, dtype=float)
    if nu_arr.ndim != 1:
        raise ValueError("nu must be a 1D array.")
    if nu_arr.shape[0] != macro_count:
        raise ValueError(f"nu length ({nu_arr.shape[0]}) must equal n_macro ({macro_count}).")
    if not np.all(np.isfinite(nu_arr)):
        raise ValueError("nu contains non-finite entries.")
    if float(nu_arr.min()) < -tol:
        raise ValueError("nu must be nonnegative within tolerance.")
    nu_arr = np.where((nu_arr < 0.0) & (nu_arr >= -tol), 0.0, nu_arr)

    proto_list = _prepare_prototypes(
        prototypes=prototypes,
        grouped=grouped,
        n_micro=n_micro,
        tol=tol,
        f=labels,
        n_macro=macro_count,
    )

    for x, idx in enumerate(grouped):
        if idx.size == 0 and nu_arr[x] > tol:
            raise ValueError(f"Cannot place positive mass on empty fiber for label {x}.")

    mu = np.zeros(n_micro, dtype=float)
    for x, proto in enumerate(proto_list):
        if nu_arr[x] == 0.0:
            continue
        mu += nu_arr[x] * proto

    _assert_mass_preserved(mu.sum(), nu_arr.sum(), tol, "U_f mass preservation failed.")
    return mu


def E_tau_f(
    mu: np.ndarray,
    P: np.ndarray,
    tau: int,
    f: np.ndarray,
    prototypes: object | None = None,
    n_macro: int | None = None,
    tol: float = 1e-15,
) -> np.ndarray:
    """Compute empirical endomap ``E_{tau,f}(mu)=U_f(Q_f(mu P^tau))``."""
    validate_kernel(P)

    mu_arr = np.asarray(mu, dtype=float)
    if mu_arr.ndim != 1:
        raise ValueError("mu must be a 1D array.")
    if not np.all(np.isfinite(mu_arr)):
        raise ValueError("mu contains non-finite entries.")

    labels = np.asarray(f)
    validate_lens(labels, n_macro=n_macro) if n_macro is not None else validate_lens(labels)
    if mu_arr.shape[0] != labels.shape[0]:
        raise ValueError(f"mu length ({mu_arr.shape[0]}) must equal lens length ({labels.shape[0]}).")

    P_tau = power(P, tau)
    mu1 = mu_arr @ P_tau
    nu1 = pushforward(mu1, labels, n_macro=n_macro)
    return U_f(nu1, labels, prototypes=prototypes, n_macro=n_macro, tol=tol)


def _prepare_prototypes(
    prototypes: object | None,
    grouped: list[np.ndarray],
    n_micro: int,
    tol: float,
    f: np.ndarray,
    n_macro: int,
) -> list[np.ndarray]:
    if prototypes is None:
        return uniform_prototypes(f, n_macro=n_macro, tol=tol)

    if isinstance(prototypes, Mapping):
        proto_list = uniform_prototypes(f, n_macro=n_macro, tol=tol)
        for key, value in prototypes.items():
            if not isinstance(key, (int, np.integer)):
                raise ValueError("Prototype dict keys must be integer macro labels.")
            label = int(key)
            if label < 0 or label >= n_macro:
                raise ValueError(f"Prototype label {label} out of range [0, {n_macro - 1}].")
            proto_list[label] = _to_global_prototype(
                value=value,
                idx=grouped[label],
                n_micro=n_micro,
                label=label,
                tol=tol,
            )
        _validate_prototypes(proto_list, grouped, n_micro, tol)
        return proto_list

    if isinstance(prototypes, Sequence) and not isinstance(prototypes, (str, bytes)):
        if len(prototypes) != n_macro:
            raise ValueError(f"Prototype sequence must have length n_macro ({n_macro}).")
        proto_list = [
            _to_global_prototype(
                value=prototypes[label],
                idx=grouped[label],
                n_micro=n_micro,
                label=label,
                tol=tol,
            )
            for label in range(n_macro)
        ]
        _validate_prototypes(proto_list, grouped, n_micro, tol)
        return proto_list

    raise ValueError("prototypes must be None, a sequence, or a dict[int, array_like].")


def _to_global_prototype(
    value: object,
    idx: np.ndarray,
    n_micro: int,
    label: int,
    tol: float,
) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Prototype for label {label} must be 1D.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Prototype for label {label} contains non-finite entries.")
    if arr.size == n_micro:
        global_proto = arr.copy()
    elif arr.size == idx.size:
        global_proto = np.zeros(n_micro, dtype=float)
        if idx.size > 0:
            global_proto[idx] = arr
    else:
        raise ValueError(
            f"Prototype for label {label} must have length {n_micro} (global) "
            f"or {idx.size} (local)."
        )

    if float(global_proto.min()) < -tol:
        raise ValueError(f"Prototype for label {label} must be nonnegative within tolerance.")
    global_proto = np.where(
        (global_proto < 0.0) & (global_proto >= -tol),
        0.0,
        global_proto,
    )
    return global_proto


def _validate_prototypes(
    proto_list: list[np.ndarray],
    grouped: list[np.ndarray],
    n_micro: int,
    tol: float,
) -> None:
    if len(proto_list) != len(grouped):
        raise ValueError("Prototype list length must equal number of macro labels.")

    all_indices = np.arange(n_micro, dtype=int)
    for label, (proto, idx) in enumerate(zip(proto_list, grouped)):
        if proto.ndim != 1 or proto.shape[0] != n_micro:
            raise ValueError(f"Prototype for label {label} must be global length {n_micro}.")
        if not np.all(np.isfinite(proto)):
            raise ValueError(f"Prototype for label {label} contains non-finite entries.")
        if float(proto.min()) < -tol:
            raise ValueError(f"Prototype for label {label} must be nonnegative within tolerance.")

        proto = np.where((proto < 0.0) & (proto >= -tol), 0.0, proto)

        outside = np.setdiff1d(all_indices, idx, assume_unique=True)
        if outside.size > 0 and np.any(np.abs(proto[outside]) > tol):
            raise ValueError(f"Prototype for label {label} has mass outside its fiber.")

        if idx.size > 0:
            total = float(proto.sum())
            if not np.isclose(total, 1.0, atol=max(tol, 1e-12), rtol=1e-12):
                raise ValueError(f"Prototype for label {label} must sum to 1.")


def _assert_mass_preserved(lhs: float, rhs: float, tol: float, message: str) -> None:
    if not np.isclose(lhs, rhs, atol=max(tol, 1e-12), rtol=1e-12):
        raise RuntimeError(message)
