import json
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_EXPERIMENTS = [
    "exp_parity_vs_and",
    "exp_parity_robustness",
    "exp_gate_phase_diagram",
    "exp_reversible_vs_erased",
    "exp_discovery_smoke",
    "exp_gate_discovery",
]

EXPECTED_ARTIFACTS = {
    "exp_parity_vs_and/summary.csv",
    "exp_parity_vs_and/config_used.json",
    "exp_parity_robustness/summary.csv",
    "exp_parity_robustness/config_used.json",
    "exp_parity_robustness/stats.json",
    "exp_gate_phase_diagram/summary.csv",
    "exp_gate_phase_diagram/config_used.json",
    "exp_gate_phase_diagram/stats.json",
    "exp_reversible_vs_erased/comparison.csv",
    "exp_reversible_vs_erased/config_used.json",
    "exp_reversible_vs_erased/stats.json",
    "exp_discovery_smoke/not_candidates.csv",
    "exp_discovery_smoke/parity_sector_candidates.csv",
    "exp_discovery_smoke/summary.json",
    "exp_gate_discovery/gates.json",
    "exp_gate_discovery/not_output_candidates.csv",
    "exp_gate_discovery/config_used.json",
}


def test_reproduce_all_manifest_deterministic(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    cmd = [
        sys.executable,
        "scripts/reproduce_all.py",
        "--results-dir",
        str(results_dir),
    ]

    subprocess.run(cmd, check=True, env=env)
    manifest_path = results_dir / "final_manifest.json"
    assert manifest_path.exists()
    first_manifest = manifest_path.read_bytes()

    subprocess.run(cmd, check=True, env=env)
    second_manifest = manifest_path.read_bytes()
    assert first_manifest == second_manifest

    for exp_name in EXPECTED_EXPERIMENTS:
        assert (results_dir / exp_name).is_dir()

    payload = json.loads(second_manifest.decode("utf-8"))
    assert payload["total_artifact_count"] == 17
    assert payload["experiments"] == EXPECTED_EXPERIMENTS

    artifacts = payload["artifacts"]
    paths = {entry["path"] for entry in artifacts}
    assert "final_manifest.json" not in paths
    assert paths == EXPECTED_ARTIFACTS

    for rel_path in EXPECTED_ARTIFACTS:
        assert (results_dir / rel_path).is_file()
