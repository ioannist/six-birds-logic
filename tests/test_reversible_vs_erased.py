import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def test_reversible_vs_erased_demo(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg_rev_vs_erased.json"
    cfg_path.write_text(
        """{
  "exp_name": "tmp_reversible_vs_erased",
  "gate_name": "cnot",
  "params": {
    "degeneracy": 3,
    "p_gate": 0.02,
    "p_mem": 0.0
  },
  "tau": 1,
  "probe_macro_input": 2
}
""",
        encoding="utf-8",
    )

    results_dir = tmp_path / "results"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    cmd = [
        sys.executable,
        "experiments/run_reversible_vs_erased.py",
        "--config",
        str(cfg_path),
        "--results-dir",
        str(results_dir),
    ]
    subprocess.run(cmd, check=True, env=env)

    out_dir = results_dir / "tmp_reversible_vs_erased"
    comparison_path = out_dir / "comparison.csv"
    config_used_path = out_dir / "config_used.json"
    stats_path = out_dir / "stats.json"

    assert comparison_path.exists()
    assert config_used_path.exists()
    assert stats_path.exists()

    with comparison_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 3
    by_view = {row["view"]: row for row in rows}
    assert set(by_view.keys()) == {"cnot_micro", "cnot_macro", "xor_erased_macro"}

    assert float(by_view["cnot_micro"]["closure_defect"]) < 1e-12
    assert float(by_view["cnot_macro"]["closure_defect"]) < 1e-12
    assert float(by_view["xor_erased_macro"]["closure_defect"]) > 0.5

    assert float(by_view["xor_erased_macro"]["unretained_input_info"]) > float(
        by_view["cnot_macro"]["unretained_input_info"]
    )

    assert float(by_view["xor_erased_macro"]["rm_view"]) > float(by_view["cnot_macro"]["rm_view"]) + 0.1

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert float(stats["ratio_unretained_input_info"]) > 5.0
