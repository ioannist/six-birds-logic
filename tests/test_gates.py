import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from emergent_logic.gates import bits_to_index, fit_gate_from_samples, predicate_stability_kernel


def test_predicate_stability_kernel_uniform_2state() -> None:
    P = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)
    pred = np.array([0, 1], dtype=int)
    stability = predicate_stability_kernel(P, pred, tau=1)
    assert_allclose(stability, 0.85, atol=1e-12)


def test_truth_table_recovery_xor_low_noise() -> None:
    tt = np.array([0, 1, 1, 0], dtype=int)
    n = 20000
    rng = np.random.default_rng(0)
    inputs = rng.integers(0, 2, size=(n, 2))
    idx = bits_to_index(inputs)
    y = tt[idx]
    p = 0.02
    y_noisy = y ^ (rng.random(n) < p)

    fit = fit_gate_from_samples(inputs, y_noisy)
    assert_array_equal(fit.table, tt)


def test_metrics_monotone_with_noise() -> None:
    tt = np.array([0, 1, 1, 0], dtype=int)
    p_list = [0.0, 0.05, 0.15]
    errs: list[float] = []
    cond_ents: list[float] = []

    for p in p_list:
        rng = np.random.default_rng(123)
        inputs = rng.integers(0, 2, size=(60000, 2))
        idx = bits_to_index(inputs)
        y = tt[idx]
        y_noisy = y ^ (rng.random(60000) < p)
        fit = fit_gate_from_samples(inputs, y_noisy)
        errs.append(fit.error_rate)
        cond_ents.append(fit.H_out_given_in)

    assert errs[1] + 1e-3 >= errs[0]
    assert errs[2] + 1e-3 >= errs[1]
    assert cond_ents[1] + 1e-3 >= cond_ents[0]
    assert cond_ents[2] + 1e-3 >= cond_ents[1]
