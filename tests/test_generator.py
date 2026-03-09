import numpy as np

from emergent_logic.gates import predicate_stability_kernel
from emergent_logic.generator import gate_error_rate_kernel, make_gate_lab
from emergent_logic.markov import validate_kernel


LOW_NOISE_PARAMS = dict(degeneracy=3, barrier=8.0, base_mem_noise=0.05, p_gate=0.01)


def test_make_gate_lab_kernels_validate() -> None:
    for gate_name in ["not", "cnot", "and"]:
        P, _, _ = make_gate_lab(gate_name, LOW_NOISE_PARAMS)
        assert validate_kernel(P) is True


def test_stable_bits_have_high_stability_low_noise() -> None:
    for gate_name in ["not", "cnot", "and"]:
        P, f_dict, meta = make_gate_lab(gate_name, LOW_NOISE_PARAMS)
        for bit_name in meta["stable_bits"]:
            pred = f_dict[bit_name]
            stability = predicate_stability_kernel(P, pred, tau=1)
            assert stability > 0.95


def test_gate_error_low_noise_below_threshold() -> None:
    for gate_name in ["not", "cnot", "and"]:
        P, f_dict, meta = make_gate_lab(gate_name, LOW_NOISE_PARAMS)
        err = gate_error_rate_kernel(
            P,
            f_dict["inputs"],
            f_dict["output"],
            meta["truth_table"],
            tau=1,
        )
        assert err < 0.05
