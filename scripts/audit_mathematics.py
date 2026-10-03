"""Independent finite/analytic audit of the manuscript's numerical claims.

This script does not import emergent_logic. Parity closure is checked with
exact rational transition matrices over all six balanced binary assignments.
Gate channels are checked against closed formulas obtained by composing
independent bit-flip noises. Phase RM is checked with rational matrices and
explicit stationary laws, independently of the stationary solver. Numerical
Shannon tables use floating point; the central eightfold ratio additionally
has a rational logarithm interval certificate.
"""

import argparse
import csv
import itertools
import json
import math
from fractions import Fraction as F
from pathlib import Path


def multiply(A, B):
    return [
        [sum((a * b for a, b in zip(row, col)), F(0)) for col in zip(*B)] for row in A
    ]


def matrix_power(A, tau):
    out = [[F(i == j) for j in range(len(A))] for i in range(len(A))]
    for _ in range(tau):
        out = multiply(out, A)
    return out


def macro_rows(P, labels):
    return [
        [
            sum((prob for prob, label in zip(row, labels) if label == x), F(0))
            for x in range(max(labels) + 1)
        ]
        for row in P
    ]


def mismatch(P, labels, weights):
    rows = macro_rows(P, labels)
    result = F(0)
    for x in range(max(labels) + 1):
        idx = [i for i, label in enumerate(labels) if label == x]
        mass = sum((weights[i] for i in idx), F(0))
        if not mass:
            continue
        mean = [
            sum((weights[i] * rows[i][y] for i in idx), F(0)) / mass
            for y in range(max(labels) + 1)
        ]
        result += sum(
            (weights[i] * sum(abs(a - b) for a, b in zip(rows[i], mean)) for i in idx),
            F(0),
        )
    return result / sum(weights)


def entropy(p):
    return -sum(float(x) * math.log2(float(x)) for x in p if x)


def log_bounds(q, terms=25):
    """Exact bounds on ln(q) for 1<=q<=2, via the positive atanh series.

    With t=(q-1)/(q+1), integrate the geometric series of 2/(1-t^2).
    After terms k=0,...,N-1, the tail is bounded above by
    2*t^(2*N+1)/((2*N+1)*(1-t^2)). All operations here are rational.
    """
    assert F(1) <= q <= F(2)
    t = (q - 1) / (q + 1)
    lower = 2 * sum((t ** (2 * k + 1) / F(2 * k + 1) for k in range(terms)), F(0))
    tail = 2 * t ** (2 * terms + 1) / (F(2 * terms + 1) * (1 - t * t))
    return lower, lower + tail


def certified_loss_ratio():
    """Rational interval proof of the published >8 loss ratio at p=1/50."""
    two_lo, two_hi = log_bounds(F(2))
    x_lo, x_hi = log_bounds(F(32, 25))
    y_lo, y_hi = log_bounds(F(49, 25))
    # ln(1/50)=ln(32/25)-6 ln2; ln(49/50)=ln(49/25)-ln2.
    A_lo, A_hi = (x_lo + 49 * y_lo) / 50, (x_hi + 49 * y_hi) / 50
    h_lo = F(11, 10) - A_hi / two_lo
    h_hi = F(11, 10) - A_lo / two_hi
    assert F(0) < h_lo < h_hi < F(1, 7)
    lo, hi = 1 + 1 / h_hi, 1 + 1 / h_lo
    assert lo > 8
    # Round outward, still in exact arithmetic, for a compact certificate.
    scale = 10**12
    scaled_lo, scaled_hi = lo * scale, hi * scale
    rounded_lo = F(scaled_lo.numerator // scaled_lo.denominator, scale)
    rounded_hi = F(-(-scaled_hi.numerator // scaled_hi.denominator), scale)
    assert rounded_lo <= lo <= hi <= rounded_hi
    return {
        "exact_rational_ratio_lower": str(rounded_lo),
        "exact_rational_ratio_upper": str(rounded_hi),
        "eightfold_inequality_certified": rounded_lo > 8,
    }


def assert_close(actual, expected, label, tol=2e-10):
    if not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > tol:
        raise AssertionError(f"{label}: {actual} != {expected}")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def parity_kernel(p):
    even = [(1 - p) / 2, F(4, 5) * p, F(1, 5) * p, (1 - p) / 2]
    odd = [F(3, 10) * p, (1 - p) / 2, (1 - p) / 2, F(7, 10) * p]
    return [even, odd, odd, even]


def audit_parity(results):
    rows = read_csv(results / "exp_parity_robustness" / "summary.csv")
    config = json.loads(
        (results / "exp_parity_robustness" / "config_used.json").read_text()
    )
    if config["params_base"] != {"degeneracy": 3}:
        raise AssertionError("This audit targets the published parity configuration.")
    p_values = [F(str(p)) for p in config["grid"]["p_leak"]]
    taus = config["grid"]["tau"]
    balanced = [
        [int(i in chosen) for i in range(4)]
        for chosen in itertools.combinations(range(4), 2)
    ]
    grids = {}
    for p, tau in itertools.product(p_values, taus):
        P = matrix_power(parity_kernel(p), tau)
        parity = [0, 1, 1, 0]
        rm = mismatch(P, parity, [F(1, 4)] * 4)
        assert rm == 0
        score_random = sorted(mismatch(P, f, [F(1, 4)] * 4) for f in balanced)
        median = (score_random[2] + score_random[3]) / 2
        assert median == p * abs(1 - 2 * p) ** (tau - 1) / 10
        assert median > rm
        and_rm = mismatch(P, [0, 0, 0, 1], [F(1, 4)] * 4)
        assert and_rm == F(2, 3) * abs(F(1, 2) - F(6, 5) * p) * abs(1 - 2 * p) ** (
            tau - 1
        )
        expected = {"parity": rm, "and": and_rm}
        subset = [r for r in rows if F(r["p_leak"]) == p and int(r["tau"]) == tau]
        if len(subset) != config["n_random"] + 2:
            raise AssertionError("Incomplete parity grid")
        for row in subset:
            if row["lens_type"] in expected:
                assert_close(row["rm"], expected[row["lens_type"]], "parity/AND RM")
            elif row["lens_type"] == "random":
                if min(abs(float(row["rm"]) - float(x)) for x in score_random) > 2e-10:
                    raise AssertionError(
                        "Random RM not among the exhaustive balanced lenses"
                    )
        sampled = sorted(float(r["rm"]) for r in subset if r["lens_type"] == "random")
        if (sampled[(len(sampled) - 1) // 2] + sampled[len(sampled) // 2]) / 2 <= 2e-10:
            raise AssertionError("No resolved sampled-median win")
        par = next(r for r in subset if r["lens_type"] == "parity")
        stability = (1 + (1 - 2 * p) ** tau) / 2
        assert_close(par["stability"], stability, "parity stability")
        assert_close(par["error"], 1 - stability, "parity error")
        grids[f"p={p},tau={tau}"] = {
            "exact_parity_rm": str(rm),
            "exact_exhaustive_median_rm": str(median),
        }
    if len(rows) != len(grids) * (config["n_random"] + 2):
        raise AssertionError("Unexpected extra parity rows")
    return grids


def gate_matrix(name, g, m):
    n = 3 if name == "and" else 2
    states = list(itertools.product((0, 1), repeat=n))

    def target(s):
        if name == "not":
            return (s[0], 1 - s[0])
        if name == "cnot":
            return (s[0], s[0] ^ s[1])
        return (s[0], s[1], s[0] & s[1])

    P = []
    for s in states:
        row = []
        for dest in states:
            prob = F(1)
            for j, (a, b) in enumerate(zip(target(s), dest)):
                noise = g if j == n - 1 else m
                prob *= (1 - noise) if a == b else noise
            row.append(prob)
        assert sum(row) == 1
        P.append(row)
    # Explicit stationary laws, with a uniform mixture on input registers.
    if name == "cnot":
        pi = [F(1, 4)] * 4
    else:
        pi = []
        for s in states:
            if name == "not":
                err = g + m - 2 * g * m
                p_one = (1 - err) if s[0] == 0 else err
                mass_inputs = F(1, 2)
            else:
                p_one = g + (1 - 2 * g) * ((1 - m) if s[0] else m) * (
                    (1 - m) if s[1] else m
                )
                mass_inputs = F(1, 4)
            pi.append(mass_inputs * (p_one if s[-1] else 1 - p_one))
    assert multiply([pi], P)[0] == pi
    return P, pi, [s[-1] for s in states]


def audit_phase(results):
    rows = read_csv(results / "exp_gate_phase_diagram" / "summary.csv")
    config = json.loads(
        (results / "exp_gate_phase_diagram" / "config_used.json").read_text()
    )
    expected_keys = set(
        itertools.product(
            config["gates"],
            config["grid"]["barrier"],
            config["grid"]["p_gate"],
            config["grid"]["tau"],
        )
    )
    actual_keys = [
        (r["gate_name"], float(r["barrier"]), float(r["p_gate"]), int(r["tau"]))
        for r in rows
    ]
    assert (
        set(actual_keys) == expected_keys
        and len(actual_keys) == len(expected_keys) == 108
    )
    for row in rows:
        name, g, m, tau = (
            row["gate_name"],
            F(row["p_gate"]),
            F(row["p_mem"]),
            int(row["tau"]),
        )
        assert_close(
            m,
            config["params_base"]["base_mem_noise"] * math.exp(-float(row["barrier"])),
            "memory-barrier parameter",
        )
        wander = (1 - (1 - 2 * m) ** (tau - 1)) / 2
        if name == "not":
            error = g + wander - 2 * g * wander
            channel_rows = [[1 - error, error], [error, 1 - error]]
            truth_error = error
        elif name == "and":
            p_one = [
                g
                + (1 - 2 * g)
                * (wander if a == 0 else 1 - wander)
                * (wander if b == 0 else 1 - wander)
                for a, b in itertools.product((0, 1), repeat=2)
            ]
            channel_rows = [[1 - p, p] for p in p_one]
            truth_error = (sum(p_one[:3]) + 1 - p_one[3]) / 4
        elif name == "cnot":
            error = (1 - (1 - 2 * g) ** tau * (1 - 2 * m) ** (tau // 2)) / 2
            channel_rows = [[1 - error, error]] * 4
            truth_error = error if tau % 2 else F(1, 2)
        else:
            raise AssertionError(f"Unknown gate {name}")
        induced = sum(1 - max(row) for row in channel_rows) / len(channel_rows)
        conditional = sum(entropy(row) for row in channel_rows) / len(channel_rows)
        assert_close(row["err_truth"], truth_error, "phase truth error")
        assert_close(row["err_induced"], induced, "phase induced error")
        assert_close(row["H_out_given_in"], conditional, "phase conditional entropy")
        P, pi, out = gate_matrix(name, g, m)
        rm = mismatch(matrix_power(P, tau), out, pi)
        assert_close(row["rm_output"], rm, "phase stationary RM")
    return {
        "complete_grid_rows": len(rows),
        "analytic_channel_checks": len(rows),
        "rational_stationary_RM_checks": len(rows),
    }


def audit_reversible(results):
    rows = {
        r["view"]: r
        for r in read_csv(results / "exp_reversible_vs_erased" / "comparison.csv")
    }
    config = json.loads(
        (results / "exp_reversible_vs_erased" / "config_used.json").read_text()
    )
    assert config["tau"] == 1 and config["params"]["p_mem"] == 0.0
    p, d = F(str(config["params"]["p_gate"])), config["params"]["degeneracy"]
    assert p == F(1, 50)
    h = entropy([p, 1 - p])
    for name in ["cnot_micro", "cnot_macro", "xor_erased_macro"]:
        r = rows[name]
        is_erased = name == "xor_erased_macro"
        hidden = math.log2(d) if name == "cnot_micro" else 0.0
        output_entropy = (1.0 if is_erased else 2.0) + hidden
        assert_close(r["H_in"], 2.0, "reversible H_in")
        assert_close(r["H_out"], output_entropy, "reversible H_out")
        assert_close(r["H_out_given_in"], h + hidden, "reversible conditional entropy")
        assert_close(
            r["I_in_out"],
            (1.0 if is_erased else 2.0) - h,
            "reversible mutual information",
        )
        assert_close(
            r["unretained_input_info"],
            h + (1.0 if is_erased else 0.0),
            "reversible loss",
        )
        assert_close(r["entropy_drop"], 2 - output_entropy, "reversible entropy drop")
        assert_close(
            r["closure_defect"], 1 - 2 * p if is_erased else 0, "reversible defect"
        )
        assert_close(r["rm_view"], 1 - 2 * p if is_erased else 0, "reversible RM")
        assert_close(r["epr_view"], 0, "reversible EPR")
    stats = json.loads(
        (results / "exp_reversible_vs_erased" / "stats.json").read_text()
    )
    assert_close(stats["ratio_unretained_input_info"], (1 + h) / h, "loss ratio")
    assert (1 + h) / h > 8
    return {
        "analytic_loss_ratio": (1 + h) / h,
        "erased_RM": str(1 - 2 * p),
        "ratio_certificate": certified_loss_ratio(),
    }


def audit_discovery(results):
    summary = json.loads((results / "exp_discovery_smoke" / "summary.json").read_text())
    assert (
        summary["not_best_agreement"] == summary["parity_sector_best_agreement"] == 1.0
    )
    config = json.loads(
        (results / "exp_gate_discovery" / "config_used.json").read_text()
    )
    gate = json.loads((results / "exp_gate_discovery" / "gates.json").read_text())[
        "not"
    ]
    d = config["lab"]["params"]["degeneracy"]
    assert config["tau"] == 1
    expected_input = [
        a for a, c in itertools.product((0, 1), repeat=2) for _ in range(d)
    ]
    expected_output = [
        c for a, c in itertools.product((0, 1), repeat=2) for _ in range(d)
    ]
    assert gate["input_partition_labels"] == expected_input
    assert gate["output_partition_labels"] == expected_output
    assert gate["exact_truth_table_bits"] == [1, 0]
    assert gate["truth_table_bits"] == [1, 0]
    p = F(str(config["lab"]["params"]["p_gate"]))
    h = entropy([p, 1 - p])
    assert_close(gate["exact_error"], p, "discovery exact error")
    assert_close(gate["exact_entropy"], h, "discovery exact entropy")
    assert_close(gate["output_future_I"], 1 - h, "discovery future information")
    assert_close(gate["output_current_I"], 0, "discovery current information")
    assert_close(gate["output_delta_I"], 1 - h, "discovery predictive gain")
    assert gate["sample_size"] == config["n_samples"] == 20000
    confusion = gate["confusion"]
    assert len(confusion) == 2 and all(len(row) == 2 for row in confusion)
    assert all(isinstance(x, int) and x >= 0 for row in confusion for x in row)
    assert [sum(row) for row in confusion] == [10000, 10000]
    empirical_error = sum(min(row) for row in confusion) / gate["sample_size"]
    empirical_entropy = (
        sum(entropy([F(x, sum(row)) for x in row]) for row in confusion) / 2
    )
    assert_close(gate["error"], empirical_error, "sample error from confusion counts")
    assert_close(
        gate["entropy"], empirical_entropy, "sample entropy from confusion counts"
    )
    # Empirical sample error is a sample statistic, distinct from the exact
    # kernel error. A generous binomial bound catches gross inconsistencies.
    se = math.sqrt(float(p * (1 - p)) / gate["sample_size"])
    assert abs(gate["error"] - float(p)) < 6 * se
    return {
        "partition_agreement": 1.0,
        "exact_gate_error": float(p),
        "exact_gate_entropy": h,
        "empirical_gate_error": gate["error"],
        "label_gauge": gate["label_convention"],
    }


def audit_sweep(results):
    rows = read_csv(results / "exp_parity_vs_and" / "summary.csv")
    config = json.loads(
        (results / "exp_parity_vs_and" / "config_used.json").read_text()
    )
    expected = set(
        itertools.product(
            config["gates"],
            config["grid"]["barrier"],
            config["grid"]["p_gate"],
            config["grid"]["tau"],
        )
    )
    actual = [
        (r["gate_name"], float(r["barrier"]), float(r["p_gate"]), int(r["tau"]))
        for r in rows
    ]
    assert set(actual) == expected and len(actual) == len(expected) == 24
    for row in rows:
        name, g, m = row["gate_name"], F(row["p_gate"]), F(row["p_mem"])
        P, _, _ = gate_matrix(name, g, m)
        P_tau = matrix_power(P, int(row["tau"]))
        n_bits = 3 if name == "and" else 2
        labels = [a ^ b for a, b, *other in itertools.product((0, 1), repeat=n_bits)]
        rm = mismatch(P_tau, labels, [F(1, len(P))] * len(P))
        R = macro_rows(P_tau, labels)
        defect = F(0)
        for x in [0, 1]:
            idx = [i for i, label in enumerate(labels) if label == x]
            mean = [sum(R[i][y] for i in idx) / len(idx) for y in [0, 1]]
            defect = max(
                defect, max(sum(abs(a - b) for a, b in zip(R[i], mean)) for i in idx)
            )
        assert_close(row["err_gate_tau1"], g, "sweep one-tick gate error")
        assert_close(row["stability_min_tau1"], 1 - m, "sweep storage stability")
        assert_close(row["rm_full"], 0, "sweep full RM")
        assert_close(row["rm_parity"], rm, "sweep parity RM")
        # Uniform input is already packaged, so the single probe vanishes
        # even when the parity lens fails; max fields audit all inputs.
        assert_close(row["comm_full"], 0, "sweep uniform full defect")
        assert_close(row["comm_parity"], 0, "sweep uniform parity defect")
        assert_close(row["comm_full_max"], 0, "sweep worst full defect")
        assert_close(row["comm_parity_max"], defect, "sweep worst parity defect")
    return {"rationally_checked_rows": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "parity": audit_parity(args.results_dir),
        "phase": audit_phase(args.results_dir),
        "reversible": audit_reversible(args.results_dir),
        "discovery": audit_discovery(args.results_dir),
        "sweep": audit_sweep(args.results_dir),
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
