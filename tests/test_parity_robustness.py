import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def test_parity_robustness_win_rate(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg_parity.json"
    cfg_path.write_text(
        """{
  "exp_name": "tmp_parity_robustness",
  "seed": 0,
  "gate_name": "parity_sector",
  "params_base": {
    "degeneracy": 3
  },
  "grid": {
    "p_leak": [0.05, 0.20],
    "tau": [1, 2]
  },
  "n_random": 20,
  "random_mode": "macro_balanced"
}
""",
        encoding="utf-8",
    )

    results_dir = tmp_path / "results"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    cmd = [
        sys.executable,
        "experiments/run_parity_robustness.py",
        "--config",
        str(cfg_path),
        "--results-dir",
        str(results_dir),
    ]
    subprocess.run(cmd, check=True, env=env)

    out_dir = results_dir / "tmp_parity_robustness"
    summary_path = out_dir / "summary.csv"
    config_used_path = out_dir / "config_used.json"
    stats_path = out_dir / "stats.json"

    assert summary_path.exists()
    assert config_used_path.exists()
    assert stats_path.exists()

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert float(stats["win_rate"]) >= 0.80

    with summary_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = set(reader.fieldnames or [])
    required = {"p_leak", "tau", "lens_type", "lens_id", "rm", "stability", "error"}
    assert required.issubset(cols)
