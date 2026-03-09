import numpy as np
from numpy.testing import assert_allclose

from emergent_logic.accounting import (
    apparent_entropy_production_rate,
    channel_from_kernel,
    channel_information_measures,
    entropy_production_rate,
)
from emergent_logic.generator import make_gate_lab


def test_entropy_production_reversible_vs_irreversible() -> None:
    P_rev = np.array(
        [
            [0.7, 0.3],
            [0.3, 0.7],
        ],
        dtype=float,
    )
    P_irr = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
        ],
        dtype=float,
    )

    epr_rev = entropy_production_rate(P_rev)
    epr_irr = entropy_production_rate(P_irr)

    assert epr_rev < 1e-12
    assert epr_irr > 1e-3


def test_apparent_epr_identity_lens_matches_micro() -> None:
    P_irr = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1],
        ],
        dtype=float,
    )

    f_full = np.array([0, 1, 2], dtype=int)
    micro = entropy_production_rate(P_irr)
    apparent = apparent_entropy_production_rate(P_irr, f_full, tau=1)
    assert_allclose(apparent, micro, atol=1e-12, rtol=1e-12)

    f_bin = np.array([0, 0, 1], dtype=int)
    app_bin = apparent_entropy_production_rate(P_irr, f_bin, tau=1)
    assert np.isfinite(app_bin)
    assert app_bin >= 0.0


def test_channel_information_deterministic_and() -> None:
    P, f_dict, _meta = make_gate_lab("and", {"degeneracy": 3, "p_gate": 0.0, "p_mem": 0.0})
    C = channel_from_kernel(P, f_dict["inputs"], f_dict["output"], tau=1)
    info = channel_information_measures(C)

    C_expected = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    assert_allclose(C, C_expected, atol=1e-12, rtol=1e-12)
    assert_allclose(info.H_in, 2.0, atol=1e-12, rtol=1e-12)
    assert_allclose(info.H_out_given_in, 0.0, atol=1e-12, rtol=1e-12)
    assert_allclose(info.H_out, 0.8112781244591328, atol=1e-12, rtol=1e-12)
    assert_allclose(info.I_in_out, info.H_out, atol=1e-12, rtol=1e-12)
    assert_allclose(info.entropy_drop, 2.0 - info.H_out, atol=1e-12, rtol=1e-12)
    assert_allclose(info.unretained_input_info, 2.0 - info.H_out, atol=1e-12, rtol=1e-12)
