"""Smoke report for generated gate labs."""

from __future__ import annotations

from emergent_logic.generator import gate_error_rate_kernel, make_gate_lab


def main() -> None:
    params = dict(degeneracy=3, barrier=8.0, base_mem_noise=0.05, p_gate=0.01)
    labels = [("not", "NOT"), ("cnot", "CNOT"), ("and", "AND")]

    for gate_name, title in labels:
        P, f_dict, _meta = make_gate_lab(gate_name, params)
        x_count = int(f_dict["full"].max()) + 1
        z_count = int(P.shape[0])
        err = gate_error_rate_kernel(P, f_dict["inputs"], f_dict["output"], _meta["truth_table"])
        print(f"{title}: |X|={x_count} |Z|={z_count} err={err:.6f}")


if __name__ == "__main__":
    main()
