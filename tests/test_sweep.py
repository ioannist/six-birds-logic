import os
import subprocess
import sys
from pathlib import Path


def test_sweep_reproducible_csv(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        """{
  "exp_name": "tmp_exp",
  "seed": 0,
  "gates": ["cnot", "and"],
  "params_base": {
    "degeneracy": 3,
    "base_mem_noise": 0.05
  },
  "grid": {
    "p_gate": [0.01],
    "barrier": [6.0],
    "tau": [1, 2]
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
        "experiments/run_sweep.py",
        "--config",
        str(cfg_path),
        "--results-dir",
        str(results_dir),
    ]

    subprocess.run(cmd, check=True, env=env)
    csv_path = results_dir / "tmp_exp" / "summary.csv"
    cfg_used_path = results_dir / "tmp_exp" / "config_used.json"
    assert csv_path.exists()
    assert cfg_used_path.exists()
    first_bytes = csv_path.read_bytes()

    subprocess.run(cmd, check=True, env=env)
    second_bytes = csv_path.read_bytes()

    assert first_bytes == second_bytes
