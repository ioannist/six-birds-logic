import json
import os
import subprocess
import sys
from pathlib import Path


REQUIRED_CHECKS = [
    "manifest_exists",
    "claims_exists",
    "parity_win_rate_threshold",
    "parity_rep_p005_beats_random",
    "parity_rep_p020_beats_random",
    "phase_row_count",
    "phase_no_anomalies",
    "phase_not_monotone",
    "phase_cnot_monotone",
    "phase_and_monotone",
    "rev_erased_ratio_threshold",
    "rev_erased_closure_delta",
    "rev_erased_rm_delta",
    "discovery_not_agreement",
    "discovery_parity_agreement",
    "not_gate_truth_table",
    "not_gate_error_threshold",
    "repo_no_build_dir",
    "repo_no_egg_info",
    "repo_no_stray_outer_lean_scaffold",
]


def test_validate_final_state(tmp_path: Path) -> None:
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
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/validate_final_state.py",
            "--results-dir",
            str(results_dir),
            "--repo-root",
            ".",
        ],
        check=False,
        cwd=repo_root,
        env=env,
    )
    assert proc.returncode == 0

    report_path = results_dir / "final_claims" / "seal_report.json"
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    checks = report["checks"]

    for name in REQUIRED_CHECKS:
        assert name in checks
        assert checks[name]["ok"] is True
