"""Minimal gate discovery utilities from discovered binary partitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emergent_logic.accounting import channel_from_kernel, channel_information_measures
from emergent_logic.discovery import best_binary_partition
from emergent_logic.markov import validate_kernel


def bidirectional_behavior_classes(
    P: np.ndarray,
    decimals: int = 12,
) -> np.ndarray:
    """
    Assign each state to a behavior class from rounded outgoing+incoming signatures.

    Signature(state i) = concat(round(P[i,:], d), round(P[:,i], d)).
    """
    validate_kernel(P)
    arr = np.asarray(P, dtype=float)
    out_sig = np.round(arr, decimals=decimals)
    in_sig = np.round(arr.T, decimals=decimals)
    sig = np.concatenate([out_sig, in_sig], axis=1)

    _uniq, inv = np.unique(sig, axis=0, return_inverse=True)
    return inv.astype(int, copy=False)


def binary_partitions_from_classes(
    class_labels: np.ndarray,
) -> list[np.ndarray]:
    """
    Enumerate nontrivial binary partitions induced by class labels.

    Canonicalization:
      - class containing state 0 is labeled 0
      - trivial all-0/all-1 partitions removed
      - duplicates removed deterministically
    """
    cls = np.asarray(class_labels)
    if cls.ndim != 1 or cls.size < 1:
        raise ValueError("class_labels must be a non-empty 1D array.")
    if not np.issubdtype(cls.dtype, np.integer):
        raise ValueError("class_labels must have integer dtype.")
    if np.any(cls < 0):
        raise ValueError("class_labels must be nonnegative.")

    uniq = np.unique(cls)
    n_classes = int(uniq.size)
    if n_classes < 2:
        return []

    class_to_pos = {int(c): i for i, c in enumerate(uniq.tolist())}
    pos_labels = np.array([class_to_pos[int(c)] for c in cls], dtype=int)
    pos0 = int(pos_labels[0])

    out: list[np.ndarray] = []
    seen: set[bytes] = set()

    for mask in range(1 << n_classes):
        if ((mask >> pos0) & 1) == 1:
            continue

        class_assign = np.array([(mask >> i) & 1 for i in range(n_classes)], dtype=int)
        labels = class_assign[pos_labels]
        if np.all(labels == 0) or np.all(labels == 1):
            continue

        key = labels.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(labels.astype(int, copy=True))

    out.sort(key=lambda x: x.tobytes())
    return out


def instantaneous_channel(
    f_input: np.ndarray,
    f_output: np.ndarray,
) -> np.ndarray:
    """
    Construct p(y_t | u_t) from two same-time labelings via uniform-in-input-state averaging.
    """
    fin = np.asarray(f_input)
    fout = np.asarray(f_output)
    if fin.ndim != 1 or fout.ndim != 1 or fin.shape[0] != fout.shape[0]:
        raise ValueError("f_input and f_output must be 1D arrays with matching length.")
    if not np.issubdtype(fin.dtype, np.integer):
        raise ValueError("f_input must have integer dtype.")
    if not np.issubdtype(fout.dtype, np.integer):
        raise ValueError("f_output must have integer dtype.")
    if np.any(fin < 0) or np.any(fout < 0):
        raise ValueError("f_input and f_output labels must be nonnegative.")

    n_inputs = int(fin.max()) + 1
    n_outputs = int(fout.max()) + 1

    c = np.zeros((n_inputs, n_outputs), dtype=float)
    for u in range(n_inputs):
        idx = np.where(fin == u)[0]
        if idx.size == 0:
            raise ValueError(f"Input label {u} has no supporting states.")
        c[u, :] = np.bincount(fout[idx], minlength=n_outputs) / float(idx.size)

    if not np.allclose(c.sum(axis=1), 1.0, atol=1e-12, rtol=1e-12):
        raise RuntimeError("instantaneous_channel rows are not stochastic.")
    return c


@dataclass(frozen=True)
class OutputBitCandidate:
    labels: np.ndarray
    truth_table_bits: np.ndarray
    truth_table_matrix_flat: np.ndarray
    future_channel: np.ndarray
    current_channel: np.ndarray
    error: float
    entropy: float
    future_I: float
    current_I: float
    delta_I: float
    size0: int
    size1: int


def discover_output_bit_for_input(
    P: np.ndarray,
    input_labels: np.ndarray,
    tau: int = 1,
    class_labels: np.ndarray | None = None,
) -> tuple[OutputBitCandidate, list[OutputBitCandidate]]:
    """
    Discover binary output partition for a known binary input partition.

    Ranking (lexicographic):
      1) descending delta_I
      2) ascending error
      3) ascending entropy
      4) descending future_I
      5) ascending |size0-size1|
    """
    validate_kernel(P)
    arr = np.asarray(P, dtype=float)
    n = arr.shape[0]

    inp = np.asarray(input_labels)
    if inp.ndim != 1 or inp.shape[0] != n:
        raise ValueError(f"input_labels must be 1D with length {n}.")
    if not np.issubdtype(inp.dtype, np.integer) and inp.dtype != np.bool_:
        raise ValueError("input_labels must have integer/bool dtype.")
    if not np.all((inp == 0) | (inp == 1)):
        raise ValueError("input_labels must be binary (0/1).")
    inp = inp.astype(int, copy=False)

    cls = bidirectional_behavior_classes(arr) if class_labels is None else np.asarray(class_labels)
    parts = binary_partitions_from_classes(cls)

    candidates: list[OutputBitCandidate] = []
    for labels in parts:
        out = labels.astype(int, copy=False)

        if np.all(out == 0) or np.all(out == 1):
            continue
        if np.array_equal(out, inp) or np.array_equal(out, 1 - inp):
            continue

        Cf = channel_from_kernel(arr, inp, out, tau=tau)
        truth_bits = Cf.argmax(axis=1).astype(int)
        if np.all(truth_bits == 0) or np.all(truth_bits == 1):
            continue

        info_f = channel_information_measures(Cf)
        error = float(np.mean(1.0 - np.max(Cf, axis=1)))
        entropy = float(info_f.H_out_given_in)
        future_I = float(info_f.I_in_out)

        C0 = instantaneous_channel(inp, out)
        info_0 = channel_information_measures(C0)
        current_I = float(info_0.I_in_out)

        delta_I = future_I - current_I
        size0 = int(np.sum(out == 0))
        size1 = int(np.sum(out == 1))

        b0 = int(truth_bits[0])
        b1 = int(truth_bits[1])
        tt_flat = np.array([1 - b0, b0, 1 - b1, b1], dtype=int)

        candidates.append(
            OutputBitCandidate(
                labels=out.astype(int, copy=True),
                truth_table_bits=truth_bits.astype(int, copy=True),
                truth_table_matrix_flat=tt_flat,
                future_channel=Cf.astype(float, copy=True),
                current_channel=C0.astype(float, copy=True),
                error=error,
                entropy=entropy,
                future_I=future_I,
                current_I=current_I,
                delta_I=delta_I,
                size0=size0,
                size1=size1,
            )
        )

    candidates.sort(
        key=lambda c: (
            -float(c.delta_I),
            float(c.error),
            float(c.entropy),
            -float(c.future_I),
            abs(int(c.size0) - int(c.size1)),
        )
    )

    if not candidates:
        raise RuntimeError("No valid output-bit candidates discovered.")
    return candidates[0], candidates


def discover_input_bit(
    P: np.ndarray,
    tau: int = 1,
):
    """Small helper wrapper for input-bit discovery via spectral candidate ranking."""
    return best_binary_partition(P, tau=tau)
