import numpy as np
import pytest

from emergent_logic.markov import power, simulate, stationary_distribution, validate_kernel


def test_validate_kernel_ok_and_failures() -> None:
    ok = np.array([[0.7, 0.3], [0.2, 0.8]])
    assert validate_kernel(ok)

    bad_row_sum = np.array([[0.7, 0.2], [0.2, 0.8]])
    assert validate_kernel(bad_row_sum, raise_on_fail=False) is False

    bad_negative = np.array([[1.1, -0.1], [0.0, 1.0]])
    assert validate_kernel(bad_negative, raise_on_fail=False) is False



def test_stationary_distribution_known_2state() -> None:
    P = np.array([[0.9, 0.1], [0.5, 0.5]])
    pi = stationary_distribution(P)
    expected = np.array([5.0 / 6.0, 1.0 / 6.0])
    np.testing.assert_allclose(pi, expected, atol=1e-6)



def test_power_basic() -> None:
    P = np.array([[0.9, 0.1], [0.5, 0.5]])
    np.testing.assert_allclose(power(P, 0), np.eye(2))
    np.testing.assert_allclose(power(P, 1), P)
    np.testing.assert_allclose(power(P, 2), P @ P)



def test_simulate_valid_indices() -> None:
    P = np.array([[0.8, 0.2], [0.1, 0.9]])
    traj = simulate(P, x0=0, steps=100, rng=np.random.default_rng(123))
    assert traj.shape == (101,)
    assert np.issubdtype(traj.dtype, np.integer)
    assert int(traj.min()) >= 0
    assert int(traj.max()) <= 1
