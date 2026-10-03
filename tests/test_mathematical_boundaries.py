"""Counterexamples and independent identities for the mathematical audit."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from emergent_logic.accounting import (
    apparent_entropy_production_rate,
    channel_information_measures,
    entropy_production_rate,
)
from emergent_logic.discovery import (
    binary_thresholds_from_vector,
    spectral_second_vector,
)
from emergent_logic.endomap import E_tau_f, U_f
from emergent_logic.gates import fit_gate_from_samples
from emergent_logic.gate_discovery import (
    discover_input_bit,
    discover_output_bit_for_input,
)
from emergent_logic.generator import make_gate_lab
from emergent_logic.lens import pushforward
from emergent_logic.markov import stationary_distribution, stationary_weights
from emergent_logic.markov import normalize_kernel, validate_kernel
from emergent_logic.metrics import (
    distribution_commutation_defect,
    induced_macro_kernel,
    micro_to_macro_rows,
    route_mismatch,
    worst_case_commutation_defect,
)


def test_periodic_chain_with_nonuniform_stationary_law():
    P = np.array([[0.0, 1.0, 0.0], [0.25, 0.0, 0.75], [0.0, 1.0, 0.0]])
    pi = stationary_distribution(P, max_iter=100)
    assert_allclose(pi, [0.125, 0.5, 0.375], atol=1e-12)
    assert_allclose(pi @ P, pi, atol=1e-12)


def test_slow_chain_cannot_pass_with_wrong_stationary_law():
    P = np.array([[1 - 1e-16, 1e-16], [1e-20, 1.0]])
    with pytest.raises(ValueError, match="stationary"):
        stationary_weights(P, np.array([0.5, 0.5]))
    with pytest.raises(RuntimeError, match="did not converge"):
        stationary_distribution(P, max_iter=20)


def test_bit_indices_reject_signed_overflow():
    from emergent_logic.gates import bits_to_index, index_to_bits

    with pytest.raises(ValueError, match="signed integer"):
        bits_to_index(np.ones(64, dtype=int))
    with pytest.raises(ValueError, match="signed integer"):
        index_to_bits(0, 64)


def test_kernel_validation_does_not_admit_negative_probabilities():
    invalid = np.array([[1.0, -1e-16], [0.0, 1.0]])
    assert not validate_kernel(invalid, raise_on_fail=False)
    assert_allclose(normalize_kernel(invalid), np.eye(2))
    assert_allclose(normalize_kernel(np.full((2, 2), 1e308)), np.full((2, 2), 0.5))


def test_epr_respects_tiny_positive_and_zero_reverse_flux():
    a, b = 0.2, 1e-16
    P = np.array([[1 - a - b, a, b], [b, 1 - a - b, a], [a, b, 1 - a - b]])
    expected = (a - b) * np.log(a / b)
    assert_allclose(entropy_production_rate(P), expected, atol=1e-12)
    tiny = 1e-16
    one_way = np.array([[1 - tiny, tiny, 0], [0, 1 - tiny, tiny], [tiny, 0, 1 - tiny]])
    assert np.isinf(entropy_production_rate(one_way))


def test_stationary_epr_rejects_nonstationary_weights():
    P = np.array([[0.9, 0.1], [0.5, 0.5]])
    with pytest.raises(ValueError, match="stationary"):
        entropy_production_rate(P, pi=np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="stationary"):
        apparent_entropy_production_rate(
            P, np.array([0, 1]), weights=np.array([0.5, 0.5])
        )


def test_apparent_epr_keeps_reducible_stationary_mixture():
    cycle = np.array([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
    P = np.zeros((5, 5))
    P[:3, :3] = cycle
    P[3:, 3:] = [[0.7, 0.3], [0.3, 0.7]]
    pi = np.array([0.8 / 3, 0.8 / 3, 0.8 / 3, 0.1, 0.1])
    assert_allclose(
        apparent_entropy_production_rate(P, np.arange(5), weights=pi),
        0.8 * entropy_production_rate(cycle),
        atol=1e-12,
    )


def test_null_fibers_and_support_only_route_agreement():
    P = np.array([[1.0, 0, 0], [0, 1, 0], [1, 0, 0]])
    pi = np.array([0.5, 0.5, 0.0])
    assert apparent_entropy_production_rate(P, np.arange(3), weights=pi) == 0
    f = np.array([0, 1, 1])
    assert route_mismatch(P, f, weights=pi) == 0
    assert_allclose(route_mismatch(P, f), 2 / 3)
    assert_allclose(worst_case_commutation_defect(P, f), 1)


def test_smoothed_mutual_information_uses_one_joint_distribution():
    fit = fit_gate_from_samples(np.array([0, 1]), np.array([0, 0]), k=1, smoothing=1.0)
    assert_allclose(fit.I_in_out, 0, atol=1e-12)
    info = channel_information_measures(np.array([[2 / 3, 1 / 3], [2 / 3, 1 / 3]]))
    assert_allclose(fit.H_out_given_in, info.H_out_given_in)
    with pytest.raises(ValueError, match="Every input"):
        fit_gate_from_samples(np.array([0, 0]), np.array([0, 1]), k=1)


def test_thresholds_do_not_split_roundoff_equal_coordinates():
    v = np.array([-1.0, -1.0 + 1e-14, 1.0, 1.0 + 1e-14])
    cuts = binary_thresholds_from_vector(v)
    assert cuts.size == 1
    assert abs(cuts[0]) < 1e-12


def test_spectral_vector_is_an_eigenfunction_for_nonuniform_reversible_law():
    P = np.array([[0.9, 0.1, 0], [0.2, 0.5, 0.3], [0, 0.3, 0.7]])
    pi = np.array([0.5, 0.25, 0.25])
    eig, v = spectral_second_vector(P, pi=pi)
    assert_allclose(P @ v, eig * v, atol=1e-12)
    assert abs(pi @ v) < 1e-12
    with pytest.raises(ValueError, match="stationary"):
        spectral_second_vector(P, pi=np.ones(3) / 3)
    with pytest.raises(ValueError, match="positive stationary"):
        spectral_second_vector(np.eye(3), pi=np.array([0.5, 0.5, 0.0]))


def test_uniform_probe_can_mask_complete_failure_of_lumpability():
    P, f, _ = make_gate_lab("cnot", {"p_gate": 0.0, "p_mem": 0.0, "degeneracy": 1})
    labels = f["output"]
    assert distribution_commutation_defect(np.ones(4) / 4, P, labels) == 0
    assert worst_case_commutation_defect(P, labels) == 1
    assert route_mismatch(P, labels) == 1
    assert worst_case_commutation_defect(P, labels, tau=2) == 0


def test_weighted_rm_has_micro_mass_normalization_and_commutation_bound():
    P = np.array([[0.8, 0.2, 0], [0.1, 0.4, 0.5], [0.2, 0.1, 0.7]])
    f = np.array([0, 0, 1])
    mu = np.array([0.1, 0.2, 0.7])
    rows = micro_to_macro_rows(P, f)
    K = induced_macro_kernel(P, f, weights=mu)
    expected = np.sum(mu * np.abs(rows - K[f]).sum(axis=1))
    assert_allclose(route_mismatch(P, f, weights=mu), expected)
    prototypes = {0: mu[:2] / mu[:2].sum(), 1: np.array([1.0])}
    assert (
        distribution_commutation_defect(mu, P, f, prototypes=prototypes)
        <= expected + 1e-12
    )


def test_probability_maps_reject_signed_input_before_evolution():
    mu, f = np.array([-1.0, 2.0]), np.array([0, 0])
    P = np.ones((2, 2)) / 2
    for operation in [
        lambda: pushforward(mu, f),
        lambda: E_tau_f(mu, P, 1, f),
        lambda: distribution_commutation_defect(mu, P, f),
    ]:
        with pytest.raises(ValueError, match="nonnegative"):
            operation()
    with pytest.raises(ValueError, match="empty fiber"):
        U_f(np.array([1.0, 1e-20]), f, n_macro=2)


def test_generator_does_not_silently_accept_unimplemented_erasure():
    with pytest.raises(ValueError, match="retains"):
        make_gate_lab("cnot", {"ancilla_mode": "erase"})
    with pytest.raises(ValueError, match="integer"):
        make_gate_lab("parity_sector", {"degeneracy": 1.5})


def test_parity_lab_input_labels_match_its_identity_truth_table():
    from emergent_logic.generator import gate_error_rate_kernel

    P, f, meta = make_gate_lab("parity_sector", {"p_leak": 0.05})
    assert meta["k_inputs"] == 1
    assert np.unique(f["inputs"]).tolist() == [0, 1]
    for tau in [1, 2, 4]:
        error = gate_error_rate_kernel(
            P, f["inputs"], f["output"], meta["truth_table"], tau=tau
        )
        assert_allclose(error, (1 - (1 - 0.1) ** tau) / 2, atol=1e-12)


def test_gate_discovery_has_negative_controls_and_explicit_label_gauge():
    P, _, _ = make_gate_lab("not", {"p_gate": 0.05, "p_mem": 0.001, "degeneracy": 1})
    inp = discover_input_bit(P).labels
    best, _ = discover_output_bit_for_input(P, inp)
    swapped, _ = discover_output_bit_for_input(P, 1 - inp)
    assert best.truth_table_bits.tolist() == [1, 0]
    assert swapped.truth_table_bits.tolist() == [0, 1]
    assert_allclose(swapped.delta_I, best.delta_I)
    assert_allclose(swapped.error, best.error)
    with pytest.raises(RuntimeError, match="predictive information gain"):
        discover_output_bit_for_input(np.eye(4), np.array([0, 0, 1, 1]))
    with pytest.raises(RuntimeError, match="No valid output"):
        discover_output_bit_for_input(np.ones((4, 4)) / 4, np.array([0, 0, 1, 1]))
