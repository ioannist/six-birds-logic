import numpy as np
from numpy.testing import assert_allclose

from emergent_logic.endomap import E_tau_f, U_f
from emergent_logic.lens import pushforward


def test_pushforward_of_lift_matches_nu_uniform() -> None:
    f = np.array([0, 0, 1, 1], dtype=int)
    nu = np.array([0.3, 0.7], dtype=float)
    mu = U_f(nu, f)
    nu2 = pushforward(mu, f)
    assert_allclose(nu2, nu, atol=1e-12)


def test_pushforward_of_lift_matches_nu_custom_prototypes() -> None:
    f = np.array([0, 0, 1, 1], dtype=int)
    nu = np.array([0.3, 0.7], dtype=float)
    prototypes = {
        0: np.array([1.0, 0.0], dtype=float),
        1: np.array([0.25, 0.75], dtype=float),
    }
    mu = U_f(nu, f, prototypes=prototypes)
    nu2 = pushforward(mu, f)
    assert_allclose(nu2, nu, atol=1e-12)


def test_endomap_identity_kernel_equals_lift_of_pushforward() -> None:
    f = np.array([0, 0, 1, 1], dtype=int)
    P = np.eye(4, dtype=float)
    mu = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    tau = 3

    out = E_tau_f(mu, P, tau, f)
    expected = U_f(pushforward(mu, f), f)
    assert_allclose(out, expected, atol=1e-12)


def test_endomap_idempotent_on_packaged_state_when_identity() -> None:
    f = np.array([0, 0, 1, 1], dtype=int)
    nu = np.array([0.35, 0.65], dtype=float)
    mu0 = U_f(nu, f)
    P = np.eye(4, dtype=float)

    out = E_tau_f(mu0, P, 5, f)
    assert_allclose(out, mu0, atol=1e-12)
