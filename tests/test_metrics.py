import numpy as np

from emergent_logic.metrics import distribution_commutation_defect, route_mismatch


def test_perfectly_lumpable_chain_has_zero_rm_and_zero_defect() -> None:
    f = np.array([0, 0, 1, 1], dtype=int)
    P_perfect = np.array(
        [
            [0.6, 0.1, 0.2, 0.1],
            [0.6, 0.1, 0.2, 0.1],
            [0.2, 0.1, 0.6, 0.1],
            [0.2, 0.1, 0.6, 0.1],
        ],
        dtype=float,
    )
    rm = route_mismatch(P_perfect, f)
    assert rm < 1e-12

    mu = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    defect = distribution_commutation_defect(mu, P_perfect, f, tau=1)
    assert defect < 1e-12


def test_broken_lumpability_increases_rm_and_defect() -> None:
    f = np.array([0, 0, 1, 1], dtype=int)
    P_perfect = np.array(
        [
            [0.6, 0.1, 0.2, 0.1],
            [0.6, 0.1, 0.2, 0.1],
            [0.2, 0.1, 0.6, 0.1],
            [0.2, 0.1, 0.6, 0.1],
        ],
        dtype=float,
    )
    P_broken = P_perfect.copy()
    P_broken[1] = np.array([0.5, 0.1, 0.25, 0.15], dtype=float)

    rm0 = route_mismatch(P_perfect, f)
    rm1 = route_mismatch(P_broken, f)
    assert rm1 > rm0 + 0.01

    mu = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    defect1 = distribution_commutation_defect(mu, P_broken, f, tau=1)
    assert defect1 > 1e-6
