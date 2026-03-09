"""Run minimal gate discovery on a NOT lab and log discovered operator artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from emergent_logic.gate_discovery import discover_input_bit, discover_output_bit_for_input
from emergent_logic.gates import fit_gate_from_samples
from emergent_logic.generator import make_gate_lab
from emergent_logic.markov import power


CAND_COLUMNS = [
    "rank",
    "delta_I",
    "future_I",
    "current_I",
    "error",
    "entropy",
    "truth_table_bits",
    "truth_table_matrix_flat",
    "size0",
    "size1",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gate discovery experiment.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config.")
    parser.add_argument("--results-dir", default="results", help="Base results directory.")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    _validate_config(cfg)

    out_dir = Path(args.results_dir) / str(cfg["exp_name"])
    out_dir.mkdir(parents=True, exist_ok=True)

    gate_name = str(cfg["lab"]["gate_name"]).lower().replace("-", "_")
    params = dict(cfg["lab"]["params"])
    tau = int(cfg["tau"])
    seed = int(cfg["seed"])
    n_samples = int(cfg["n_samples"])

    P, _f_dict, _meta = make_gate_lab(gate_name, params)

    input_cand = discover_input_bit(P, tau=tau)
    input_labels = input_cand.labels

    output_best, output_all = discover_output_bit_for_input(P, input_labels, tau=tau)

    inputs, outputs = _sample_balanced_transitions(
        P,
        input_labels=input_labels,
        output_labels=output_best.labels,
        tau=tau,
        n_samples=n_samples,
        seed=seed,
    )

    fit = fit_gate_from_samples(inputs, outputs, k=1)
    tt_bits = fit.table.astype(int)
    b0 = int(tt_bits[0])
    b1 = int(tt_bits[1])
    tt_flat = [1 - b0, b0, 1 - b1, b1]

    gates_payload = {
        "not": {
            "truth_table_bits": [int(x) for x in tt_bits.tolist()],
            "truth_table_matrix_flat": tt_flat,
            "error": float(fit.error_rate),
            "entropy": float(fit.H_out_given_in),
            "sample_size": int(n_samples),
            "input_partition_score": float(input_cand.score),
            "output_delta_I": float(output_best.delta_I),
            "output_future_I": float(output_best.future_I),
            "output_current_I": float(output_best.current_I),
        }
    }

    with (out_dir / "gates.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(gates_payload, fh, sort_keys=True, indent=2)
        fh.write("\n")

    with (out_dir / "config_used.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(cfg, fh, sort_keys=True, indent=2)
        fh.write("\n")

    with (out_dir / "not_output_candidates.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CAND_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for rank, cand in enumerate(output_all, start=1):
            writer.writerow(
                {
                    "rank": _fmt(rank),
                    "delta_I": _fmt(cand.delta_I),
                    "future_I": _fmt(cand.future_I),
                    "current_I": _fmt(cand.current_I),
                    "error": _fmt(cand.error),
                    "entropy": _fmt(cand.entropy),
                    "truth_table_bits": json.dumps([int(x) for x in cand.truth_table_bits.tolist()]),
                    "truth_table_matrix_flat": json.dumps(
                        [int(x) for x in cand.truth_table_matrix_flat.tolist()]
                    ),
                    "size0": _fmt(cand.size0),
                    "size1": _fmt(cand.size1),
                }
            )


def _sample_balanced_transitions(
    P: np.ndarray,
    input_labels: np.ndarray,
    output_labels: np.ndarray,
    tau: int,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2.")

    rng = np.random.default_rng(seed)
    arr = np.asarray(P, dtype=float)
    p_tau = power(arr, tau)

    inp = np.asarray(input_labels, dtype=int)
    out = np.asarray(output_labels, dtype=int)
    if inp.ndim != 1 or out.ndim != 1 or inp.shape[0] != out.shape[0]:
        raise ValueError("input_labels and output_labels must be 1D with same length.")
    if not np.all((inp == 0) | (inp == 1)) or not np.all((out == 0) | (out == 1)):
        raise ValueError("input_labels and output_labels must be binary.")

    idx0 = np.where(inp == 0)[0]
    idx1 = np.where(inp == 1)[0]
    if idx0.size == 0 or idx1.size == 0:
        raise ValueError("input_labels must contain both classes.")

    n0 = n_samples // 2
    n1 = n_samples - n0
    start0 = rng.choice(idx0, size=n0, replace=True)
    start1 = rng.choice(idx1, size=n1, replace=True)

    starts = np.concatenate([start0, start1])
    inputs = inp[starts]

    probs = p_tau[starts]
    cdf = np.cumsum(probs, axis=1)
    r = rng.random(starts.shape[0])
    next_states = np.sum(cdf < r[:, None], axis=1)
    outputs = out[next_states]

    return inputs.astype(int, copy=False), outputs.astype(int, copy=False)


def _load_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as fh:
        if suffix == ".json":
            cfg = json.load(fh)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise RuntimeError("Install pyyaml or use JSON config.") from exc
            cfg = yaml.safe_load(fh)
        else:
            raise ValueError("Unsupported config format. Use .json, .yaml, or .yml.")
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be an object.")
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    for key in ["exp_name", "seed", "tau", "n_samples", "lab"]:
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")
    if not isinstance(cfg["seed"], int):
        raise ValueError("seed must be an integer.")
    if not isinstance(cfg["tau"], int) or int(cfg["tau"]) < 0:
        raise ValueError("tau must be an integer >= 0.")
    if not isinstance(cfg["n_samples"], int) or int(cfg["n_samples"]) < 2:
        raise ValueError("n_samples must be an integer >= 2.")

    lab = cfg["lab"]
    if not isinstance(lab, dict) or "gate_name" not in lab or "params" not in lab:
        raise ValueError("lab must contain gate_name and params.")
    gate_name = str(lab["gate_name"]).lower().replace("-", "_")
    if gate_name != "not":
        raise ValueError("This runner currently supports lab.gate_name='not' only.")
    if not isinstance(lab["params"], dict):
        raise ValueError("lab.params must be an object.")


def _fmt(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return format(x, ".12g")
    return str(value)


if __name__ == "__main__":
    main()
