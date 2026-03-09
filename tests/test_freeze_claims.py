import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def test_freeze_claims_bundle(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = tmp_path / "results"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    subprocess.run(
        [sys.executable, "scripts/reproduce_all.py", "--results-dir", str(results_dir)],
        check=True,
        cwd=repo_root,
        env=env,
    )
    subprocess.run(
        [sys.executable, "scripts/freeze_claims.py", "--results-dir", str(results_dir)],
        check=True,
        cwd=repo_root,
        env=env,
    )

    out_dir = results_dir / "final_claims"
    expected_outputs = [
        "claims.json",
        "figure_parity_robustness.csv",
        "figure_gate_phase.csv",
        "table_reversible_vs_erased.csv",
        "table_partition_discovery.csv",
        "table_gate_discovery.csv",
    ]
    for name in expected_outputs:
        assert (out_dir / name).is_file()

    claims = json.loads((out_dir / "claims.json").read_text(encoding="utf-8"))

    parity_stats = json.loads(
        (results_dir / "exp_parity_robustness" / "stats.json").read_text(encoding="utf-8")
    )
    phase_stats = json.loads(
        (results_dir / "exp_gate_phase_diagram" / "stats.json").read_text(encoding="utf-8")
    )
    rev_stats = json.loads(
        (results_dir / "exp_reversible_vs_erased" / "stats.json").read_text(encoding="utf-8")
    )
    discovery_summary = json.loads(
        (results_dir / "exp_discovery_smoke" / "summary.json").read_text(encoding="utf-8")
    )
    gates = json.loads((results_dir / "exp_gate_discovery" / "gates.json").read_text(encoding="utf-8"))

    required_claim_keys = [
        "parity_win_rate",
        "parity_rep_p005_tau1_rm",
        "random_rep_p005_tau1_median_rm",
        "parity_rep_p020_tau1_rm",
        "random_rep_p020_tau1_median_rm",
        "phase_n_rows",
        "phase_anomaly_count",
        "phase_not_err_truth_p0",
        "phase_not_err_truth_p01",
        "phase_cnot_err_truth_p0",
        "phase_cnot_err_truth_p01",
        "phase_and_err_truth_p0",
        "phase_and_err_truth_p01",
        "rev_vs_erased_ratio_unretained_input_info",
        "rev_vs_erased_closure_defect_delta",
        "rev_vs_erased_rm_delta",
        "discovery_not_best_agreement",
        "discovery_parity_best_agreement",
        "not_gate_truth_table_matrix_flat",
        "not_gate_error",
        "not_gate_entropy",
        "not_gate_sample_size",
    ]
    for key in required_claim_keys:
        assert key in claims

    rep005 = parity_stats["representative"]["p_leak=0.05,tau=1"]
    rep020 = parity_stats["representative"]["p_leak=0.2,tau=1"]
    assert_close(claims["parity_win_rate"], parity_stats["win_rate"])
    assert_close(claims["parity_rep_p005_tau1_rm"], rep005["rm_parity"])
    assert_close(claims["random_rep_p005_tau1_median_rm"], rep005["rm_median_random"])
    assert_close(claims["parity_rep_p020_tau1_rm"], rep020["rm_parity"])
    assert_close(claims["random_rep_p020_tau1_median_rm"], rep020["rm_median_random"])

    per_gate = phase_stats["per_gate"]
    assert claims["phase_n_rows"] == phase_stats["n_rows"]
    assert claims["phase_anomaly_count"] == len(phase_stats.get("anomalies", []))
    assert_close(claims["phase_not_err_truth_p0"], per_gate["not"]["mean_err_truth_pgate_0"])
    assert_close(claims["phase_not_err_truth_p01"], per_gate["not"]["mean_err_truth_pgate_0.1"])
    assert_close(claims["phase_cnot_err_truth_p0"], per_gate["cnot"]["mean_err_truth_pgate_0"])
    assert_close(claims["phase_cnot_err_truth_p01"], per_gate["cnot"]["mean_err_truth_pgate_0.1"])
    assert_close(claims["phase_and_err_truth_p0"], per_gate["and"]["mean_err_truth_pgate_0"])
    assert_close(claims["phase_and_err_truth_p01"], per_gate["and"]["mean_err_truth_pgate_0.1"])

    assert_close(
        claims["rev_vs_erased_ratio_unretained_input_info"],
        rev_stats["ratio_unretained_input_info"],
    )
    assert_close(claims["rev_vs_erased_closure_defect_delta"], rev_stats["closure_defect_delta"])
    assert_close(claims["rev_vs_erased_rm_delta"], rev_stats["rm_delta"])

    assert_close(claims["discovery_not_best_agreement"], discovery_summary["not_best_agreement"])
    assert_close(
        claims["discovery_parity_best_agreement"],
        discovery_summary["parity_sector_best_agreement"],
    )

    not_gate = gates["not"]
    assert claims["not_gate_truth_table_matrix_flat"] == not_gate["truth_table_matrix_flat"]
    assert_close(claims["not_gate_error"], not_gate["error"])
    assert_close(claims["not_gate_entropy"], not_gate["entropy"])
    assert claims["not_gate_sample_size"] == not_gate["sample_size"]

    assert row_count(out_dir / "figure_parity_robustness.csv") == 12
    assert row_count(out_dir / "figure_gate_phase.csv") == 108
    assert row_count(out_dir / "table_reversible_vs_erased.csv") == 3
    assert row_count(out_dir / "table_partition_discovery.csv") == 2
    assert row_count(out_dir / "table_gate_discovery.csv") == 1

    assert claims["not_gate_truth_table_matrix_flat"] == [0, 1, 1, 0]
    assert float(claims["not_gate_error"]) < 0.10
    assert int(claims["phase_anomaly_count"]) == 0
    assert float(claims["parity_win_rate"]) >= 0.80


def row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def assert_close(a: float, b: float, tol: float = 1e-12) -> None:
    assert math.isfinite(float(a))
    assert math.isfinite(float(b))
    assert abs(float(a) - float(b)) <= tol
