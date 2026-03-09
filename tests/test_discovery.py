import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from emergent_logic.discovery import best_binary_partition, partition_agreement
from emergent_logic.generator import make_gate_lab


def test_best_partition_recovers_not_lab() -> None:
    P, f_dict, _meta = make_gate_lab(
        "not",
        {"degeneracy": 3, "barrier": 8.0, "base_mem_noise": 0.05, "p_gate": 0.01},
    )
    cand = best_binary_partition(P, tau=1)
    agreement = partition_agreement(cand.labels, f_dict["a"])
    assert agreement >= 0.95


def test_best_partition_recovers_parity_lab() -> None:
    P, f_dict, _meta = make_gate_lab("parity_sector", {"degeneracy": 3, "p_leak": 0.05})
    cand = best_binary_partition(P, tau=1)
    agreement = partition_agreement(cand.labels, f_dict["parity"])
    assert agreement >= 0.95


def test_discovery_smoke_logs_results(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    cmd = [
        sys.executable,
        "experiments/run_discovery_smoke.py",
        "--results-dir",
        str(results_dir),
    ]
    subprocess.run(cmd, check=True, env=env)

    base = results_dir / "exp_discovery_smoke"
    not_csv = base / "not_candidates.csv"
    parity_csv = base / "parity_sector_candidates.csv"
    summary_json = base / "summary.json"

    assert not_csv.exists()
    assert parity_csv.exists()
    assert summary_json.exists()

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert float(summary["not_best_agreement"]) >= 0.95
    assert float(summary["parity_sector_best_agreement"]) >= 0.95

    for csv_path in [not_csv, parity_csv]:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            cols = set(reader.fieldnames or [])
        required = {
            "rank",
            "lab_name",
            "threshold",
            "eigenvalue2",
            "score",
            "metastability",
            "rm",
            "size0",
            "size1",
            "agreement_to_truth",
        }
        assert required.issubset(cols)
