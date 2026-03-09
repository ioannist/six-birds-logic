import json
import os
import subprocess
import sys
from pathlib import Path

from emergent_logic.gate_discovery import discover_input_bit, discover_output_bit_for_input
from emergent_logic.generator import make_gate_lab


def test_not_gate_discovery_exact_ranking() -> None:
    P, _f_dict, _meta = make_gate_lab(
        "not",
        {"degeneracy": 3, "barrier": 6.0, "base_mem_noise": 0.05, "p_gate": 0.05},
    )
    input_cand = discover_input_bit(P, tau=1)
    best, _all_candidates = discover_output_bit_for_input(P, input_cand.labels, tau=1)

    assert best.truth_table_bits.tolist() == [1, 0]
    assert best.truth_table_matrix_flat.tolist() == [0, 1, 1, 0]
    assert best.error < 0.10
    assert best.delta_I > 0.5


def test_run_gate_discovery_writes_json(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg_gate_discovery.json"
    cfg_path.write_text(
        """{
  "exp_name": "exp_gate_discovery",
  "seed": 0,
  "tau": 1,
  "n_samples": 20000,
  "lab": {
    "gate_name": "not",
    "params": {
      "degeneracy": 3,
      "barrier": 6.0,
      "base_mem_noise": 0.05,
      "p_gate": 0.05
    }
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
        "experiments/run_gate_discovery.py",
        "--config",
        str(cfg_path),
        "--results-dir",
        str(results_dir),
    ]
    subprocess.run(cmd, check=True, env=env)

    out_dir = results_dir / "exp_gate_discovery"
    gates_path = out_dir / "gates.json"
    cand_path = out_dir / "not_output_candidates.csv"
    cfg_used_path = out_dir / "config_used.json"

    assert gates_path.exists()
    assert cand_path.exists()
    assert cfg_used_path.exists()

    data = json.loads(gates_path.read_text(encoding="utf-8"))
    assert data["not"]["truth_table_matrix_flat"] == [0, 1, 1, 0]
    assert float(data["not"]["error"]) < 0.10
    assert int(data["not"]["sample_size"]) == 20000
