"""Gate-lab Markov kernel generator and exact gate error diagnostics."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from emergent_logic.gates import bits_to_index, index_to_bits, predicate_stability_kernel
from emergent_logic.lens import pushforward
from emergent_logic.markov import validate_kernel


def make_gate_lab(
    gate_name: str,
    params: dict | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """
    Build a finite Markov kernel implementing one gate tick on a micro-lifted state space.

    Returns ``(P, f_dict, meta)`` where ``P`` acts on microstates, ``f_dict`` contains
    useful lenses (micro -> macro labels), and ``meta`` stores gate metadata and
    resolved parameters.
    """
    name = gate_name.strip().lower().replace("-", "_")

    if name == "parity_sector":
        params_resolved = _resolve_parity_params(params)
        p_leak = float(params_resolved["p_leak"])
        d = int(params_resolved["degeneracy"])
        n_macro = 4
        n_micro = n_macro * d

        macro_kernel = _parity_sector_macro_kernel(
            p_leak=p_leak,
            leak_bias_even_to_01=float(params_resolved["leak_bias_even_to_01"]),
            leak_bias_odd_to_11=float(params_resolved["leak_bias_odd_to_11"]),
        )
        P = np.zeros((n_micro, n_micro), dtype=float)
        for x in range(n_macro):
            row_macro = macro_kernel[x]
            row_micro = np.repeat(row_macro / d, d)
            for g in range(d):
                z = x * d + g
                P[z, :] = row_micro

        validate_kernel(P)

        f_full = np.repeat(np.arange(n_macro, dtype=int), d)
        bits_micro = index_to_bits(f_full, 2)
        a = bits_micro[:, 0].astype(int, copy=False)
        b = bits_micro[:, 1].astype(int, copy=False)
        parity = (a ^ b).astype(int, copy=False)

        f_dict: dict[str, np.ndarray] = {
            "full": f_full.astype(int, copy=False),
            "a": a,
            "b": b,
            "parity": parity,
            "inputs": bits_to_index(np.column_stack((a, b))).astype(int, copy=False),
            "output": parity,
            "ab": f_full.astype(int, copy=False),
        }
        meta = {
            "gate_name": "parity_sector",
            "bit_names": ["a", "b"],
            "stable_bits": ["parity"],
            "k_inputs": 1,
            "truth_table": np.array([0, 1], dtype=int),
            "params_resolved": params_resolved,
        }
        return P, f_dict, meta

    params_resolved = _resolve_params(params)

    if name == "not":
        bit_names = ["a", "c"]
        stable_idx = [0]
        output_idx = 1
        k_inputs = 1
        truth_table = np.array([1, 0], dtype=int)
        update_fn: Callable[[np.ndarray], np.ndarray] = _update_not
    elif name == "cnot":
        bit_names = ["a", "b"]
        stable_idx = [0]
        output_idx = 1
        k_inputs = 2
        truth_table = np.array([0, 1, 1, 0], dtype=int)
        update_fn = _update_cnot
    elif name == "and":
        bit_names = ["a", "b", "c"]
        stable_idx = [0, 1]
        output_idx = 2
        k_inputs = 2
        truth_table = np.array([0, 0, 0, 1], dtype=int)
        update_fn = _update_and
    else:
        raise ValueError("gate_name must be one of {'not', 'cnot', 'and', 'parity_sector'}.")

    n_bits = len(bit_names)
    n_macro = 1 << n_bits
    d = params_resolved["degeneracy"]
    n_micro = n_macro * d

    macro_bits = index_to_bits(np.arange(n_macro, dtype=int), n_bits)
    macro_kernel = np.zeros((n_macro, n_macro), dtype=float)
    for x in range(n_macro):
        bits = macro_bits[x]
        bits_det = update_fn(bits)
        row = _destination_distribution(
            bits_det=bits_det,
            stable_idx=stable_idx,
            output_idx=output_idx,
            p_mem=params_resolved["p_mem"],
            p_gate=params_resolved["p_gate"],
        )
        macro_kernel[x, :] = row

    P = np.zeros((n_micro, n_micro), dtype=float)
    for x in range(n_macro):
        row_macro = macro_kernel[x]
        row_micro = np.repeat(row_macro / d, d)
        for g in range(d):
            z = x * d + g
            P[z, :] = row_micro

    validate_kernel(P)

    f_full = np.repeat(np.arange(n_macro, dtype=int), d)
    bits_micro = macro_bits[f_full]
    f_dict: dict[str, np.ndarray] = {"full": f_full.astype(int, copy=False)}

    a = bits_micro[:, 0].astype(int, copy=False)
    f_dict["a"] = a

    if name == "not":
        c = bits_micro[:, 1].astype(int, copy=False)
        f_dict["inputs"] = a
        f_dict["output"] = c
        f_dict["c"] = c
        f_dict["ac"] = f_dict["full"]
    elif name == "cnot":
        b = bits_micro[:, 1].astype(int, copy=False)
        ab = bits_to_index(np.column_stack((a, b))).astype(int, copy=False)
        f_dict["inputs"] = ab
        f_dict["output"] = b
        f_dict["b"] = b
        f_dict["parity"] = (a ^ b).astype(int, copy=False)
        f_dict["ab"] = f_dict["full"]
    else:  # and
        b = bits_micro[:, 1].astype(int, copy=False)
        c = bits_micro[:, 2].astype(int, copy=False)
        ab = bits_to_index(np.column_stack((a, b))).astype(int, copy=False)
        f_dict["inputs"] = ab
        f_dict["output"] = c
        f_dict["b"] = b
        f_dict["c"] = c
        f_dict["parity"] = (a ^ b).astype(int, copy=False)
        f_dict["ab"] = ab
        f_dict["abc"] = f_dict["full"]

    meta = {
        "gate_name": name,
        "bit_names": bit_names,
        "stable_bits": [bit_names[i] for i in stable_idx],
        "k_inputs": k_inputs,
        "truth_table": truth_table,
        "params_resolved": params_resolved,
    }
    return P, f_dict, meta


def gate_error_rate_kernel(
    P: np.ndarray,
    f_inputs: np.ndarray,
    f_output: np.ndarray,
    truth_table: np.ndarray,
    tau: int = 1,
) -> float:
    """
    Compute uniform-input average Pr[output_{t+tau} != truth_table(inputs_t)] from kernel ``P``.
    """
    validate_kernel(P)
    if not isinstance(tau, (int, np.integer)) or int(tau) < 0:
        raise ValueError("tau must be an integer >= 0.")
    tau_int = int(tau)

    n_micro = P.shape[0]

    fin = np.asarray(f_inputs)
    if fin.ndim != 1 or fin.shape[0] != n_micro:
        raise ValueError(f"f_inputs must be 1D with length {n_micro}.")
    if not np.issubdtype(fin.dtype, np.integer):
        raise ValueError("f_inputs must have integer dtype.")
    if np.any(fin < 0):
        raise ValueError("f_inputs must be nonnegative.")

    fout = np.asarray(f_output)
    if fout.ndim != 1 or fout.shape[0] != n_micro:
        raise ValueError(f"f_output must be 1D with length {n_micro}.")
    if not np.issubdtype(fout.dtype, np.integer) and fout.dtype != np.bool_:
        raise ValueError("f_output must have integer/bool dtype.")
    if not np.all((fout == 0) | (fout == 1)):
        raise ValueError("f_output values must be in {0,1}.")
    fout = fout.astype(int, copy=False)

    tt = np.asarray(truth_table)
    if tt.ndim != 1 or tt.size < 1:
        raise ValueError("truth_table must be a non-empty 1D array.")
    if not np.issubdtype(tt.dtype, np.integer) and tt.dtype != np.bool_:
        raise ValueError("truth_table must have integer/bool dtype.")
    if not np.all((tt == 0) | (tt == 1)):
        raise ValueError("truth_table values must be in {0,1}.")
    tt = tt.astype(int, copy=False)

    n_inputs = int(tt.size)
    if np.any(fin >= n_inputs):
        raise ValueError("f_inputs contains labels outside truth_table range.")

    P_tau = _matrix_power(P, tau_int)

    err = 0.0
    for u in range(n_inputs):
        idx = np.where(fin == u)[0]
        if idx.size == 0:
            raise ValueError(f"No microstates found for input pattern {u}.")
        mu = np.zeros(n_micro, dtype=float)
        mu[idx] = 1.0 / float(idx.size)
        mu_next = mu @ P_tau
        out_dist = pushforward(mu_next, fout, n_macro=2)
        target = tt[u]
        err += float(out_dist[1 - target])
    return err / float(n_inputs)


def _resolve_params(params: dict | None) -> dict:
    raw = {} if params is None else dict(params)

    degeneracy = int(raw.get("degeneracy", 3))
    if degeneracy < 1:
        raise ValueError("degeneracy must be >= 1.")

    p_gate = float(raw.get("p_gate", 0.01))
    if not (0.0 <= p_gate <= 1.0):
        raise ValueError("p_gate must be in [0,1].")

    barrier = float(raw.get("barrier", 6.0))
    base_mem_noise = float(raw.get("base_mem_noise", 0.05))
    if base_mem_noise < 0.0:
        raise ValueError("base_mem_noise must be >= 0.")

    if "p_mem" in raw:
        p_mem = float(raw["p_mem"])
    else:
        p_mem = base_mem_noise * math.exp(-barrier)
    if not (0.0 <= p_mem <= 1.0):
        raise ValueError("resolved p_mem must be in [0,1].")

    ancilla_mode = str(raw.get("ancilla_mode", "retain"))
    if ancilla_mode not in {"retain", "erase"}:
        raise ValueError("ancilla_mode must be one of {'retain', 'erase'}.")

    return {
        "degeneracy": degeneracy,
        "p_gate": p_gate,
        "p_mem": p_mem,
        "barrier": barrier,
        "base_mem_noise": base_mem_noise,
        "ancilla_mode": ancilla_mode,
    }


def _resolve_parity_params(params: dict | None) -> dict:
    raw = {} if params is None else dict(params)

    degeneracy = int(raw.get("degeneracy", 3))
    if degeneracy < 1:
        raise ValueError("degeneracy must be >= 1.")

    if "p_leak" in raw:
        p_leak = float(raw["p_leak"])
    elif "p_gate" in raw:
        p_leak = float(raw["p_gate"])
    else:
        p_leak = 0.05
    if not (0.0 <= p_leak <= 1.0):
        raise ValueError("p_leak must be in [0,1].")

    leak_bias_even_to_01 = float(raw.get("leak_bias_even_to_01", 0.8))
    leak_bias_odd_to_11 = float(raw.get("leak_bias_odd_to_11", 0.7))
    if not (0.0 <= leak_bias_even_to_01 <= 1.0):
        raise ValueError("leak_bias_even_to_01 must be in [0,1].")
    if not (0.0 <= leak_bias_odd_to_11 <= 1.0):
        raise ValueError("leak_bias_odd_to_11 must be in [0,1].")

    return {
        "degeneracy": degeneracy,
        "p_leak": p_leak,
        "leak_bias_even_to_01": leak_bias_even_to_01,
        "leak_bias_odd_to_11": leak_bias_odd_to_11,
    }


def _parity_sector_macro_kernel(
    p_leak: float,
    leak_bias_even_to_01: float,
    leak_bias_odd_to_11: float,
) -> np.ndarray:
    """
    Build a parity-favoring 4-state macro kernel over states [00, 01, 10, 11].

    Parity leak probability is exactly ``p_leak`` and parity sectors are the intended stable variable.
    """
    k = np.zeros((4, 4), dtype=float)

    same = (1.0 - p_leak) * 0.5

    # Parity-0 states (00 and 11): leak to odd states with configurable asymmetry.
    to_01 = p_leak * leak_bias_even_to_01
    to_10 = p_leak * (1.0 - leak_bias_even_to_01)
    row_even = np.array([same, to_01, to_10, same], dtype=float)

    # Parity-1 states (01 and 10): leak to even states with configurable asymmetry.
    to_11 = p_leak * leak_bias_odd_to_11
    to_00 = p_leak * (1.0 - leak_bias_odd_to_11)
    row_odd = np.array([to_00, same, same, to_11], dtype=float)

    k[0] = row_even  # 00
    k[3] = row_even  # 11
    k[1] = row_odd   # 01
    k[2] = row_odd   # 10
    return k


def _update_not(bits: np.ndarray) -> np.ndarray:
    a = int(bits[0])
    return np.array([a, 1 - a], dtype=int)


def _update_cnot(bits: np.ndarray) -> np.ndarray:
    a = int(bits[0])
    b = int(bits[1])
    return np.array([a, a ^ b], dtype=int)


def _update_and(bits: np.ndarray) -> np.ndarray:
    a = int(bits[0])
    b = int(bits[1])
    return np.array([a, b, a & b], dtype=int)


def _destination_distribution(
    bits_det: np.ndarray,
    stable_idx: list[int],
    output_idx: int,
    p_mem: float,
    p_gate: float,
) -> np.ndarray:
    n_bits = int(bits_det.shape[0])
    out = np.zeros(1 << n_bits, dtype=float)

    for flip_out in (0, 1):
        p_out = (1.0 - p_gate) if flip_out == 0 else p_gate
        base = bits_det.copy()
        if flip_out == 1:
            base[output_idx] ^= 1

        m = len(stable_idx)
        for mask in range(1 << m):
            bits = base.copy()
            prob = p_out
            for j, pos in enumerate(stable_idx):
                if (mask >> j) & 1:
                    bits[pos] ^= 1
                    prob *= p_mem
                else:
                    prob *= (1.0 - p_mem)
            out[int(bits_to_index(bits))] += prob

    total = float(out.sum())
    if total <= 0.0:
        raise RuntimeError("Degenerate destination distribution.")
    return out / total


def _matrix_power(P: np.ndarray, tau: int) -> np.ndarray:
    if tau == 0:
        return np.eye(P.shape[0], dtype=float)
    result = np.eye(P.shape[0], dtype=float)
    base = np.asarray(P, dtype=float)
    exp = int(tau)
    while exp > 0:
        if exp & 1:
            result = result @ base
        base = base @ base
        exp >>= 1
    return result
