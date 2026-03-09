import json
import os
import subprocess
import sys
from pathlib import Path


def test_gate_phase_diagram_small_config(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg_phase.json"
    cfg_path.write_text(
        """{
  "exp_name": "tmp_gate_phase",
  "seed": 0,
  "gates": ["not"],
  "params_base": {
    "degeneracy": 3,
    "base_mem_noise": 0.05
  },
  "grid": {
    "p_gate": [0.0, 0.1],
    "barrier": [0.0, 6.0],
    "tau": [1]
  }
}
""",
        encoding="utf-8",
    )

    results_dir = tmp_path / "results"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    cmd = [
        sys.executable,
        "experiments/run_gate_phase_diagram.py",
        "--config",
        str(cfg_path),
        "--results-dir",
        str(results_dir),
    ]
    subprocess.run(cmd, check=True, env=env)

    out_dir = results_dir / "tmp_gate_phase"
    summary_path = out_dir / "summary.csv"
    cfg_used_path = out_dir / "config_used.json"
    stats_path = out_dir / "stats.json"

    assert summary_path.exists()
    assert cfg_used_path.exists()
    assert stats_path.exists()

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert int(stats["n_rows"]) == 4

    per_not = stats["per_gate"]["not"]
    assert float(per_not["mean_err_truth_pgate_0.1"]) >= float(per_not["mean_err_truth_pgate_0"])
