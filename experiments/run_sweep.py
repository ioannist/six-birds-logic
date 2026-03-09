"""Deterministic experiment sweep runner for gate-lab diagnostics."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from emergent_logic.gates import predicate_stability_kernel
from emergent_logic.generator import gate_error_rate_kernel, make_gate_lab
from emergent_logic.metrics import distribution_commutation_defect, route_mismatch


REQUIRED_COLUMNS = [
    "gate_name",
    "degeneracy",
    "barrier",
    "p_gate",
    "p_mem",
    "tau",
    "err_gate_tau1",
    "stability_min_tau1",
    "rm_full",
    "rm_parity",
    "comm_full",
    "comm_parity",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic sweep over gate-lab parameters.")
    parser.add_argument("--config", required=True, help="Path to JSON or YAML config.")
    parser.add_argument("--results-dir", default="results", help="Base results directory.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_config(cfg_path)
    _validate_config(cfg)

    rows = _run_sweep(cfg)

    exp_name = str(cfg["exp_name"])
    out_dir = Path(args.results_dir) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "summary.csv"
    _write_summary_csv(csv_path, rows)

    config_used_path = out_dir / "config_used.json"
    with config_used_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(cfg, fh, sort_keys=True, indent=2)
        fh.write("\n")


def _load_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as fh:
        if suffix == ".json":
            loaded = json.load(fh)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise RuntimeError("Install pyyaml or use JSON config.") from exc
            loaded = yaml.safe_load(fh)
        else:
            raise ValueError("Unsupported config format. Use .json, .yaml, or .yml.")

    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a JSON/YAML object.")
    return loaded


def _validate_config(cfg: dict[str, Any]) -> None:
    required = ["exp_name", "seed", "gates", "params_base", "grid"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")

    if not isinstance(cfg["exp_name"], str) or not cfg["exp_name"]:
        raise ValueError("exp_name must be a non-empty string.")
    if not isinstance(cfg["seed"], int):
        raise ValueError("seed must be an integer.")
    if not isinstance(cfg["gates"], list) or len(cfg["gates"]) < 1:
        raise ValueError("gates must be a non-empty list.")
    if not all(isinstance(g, str) for g in cfg["gates"]):
        raise ValueError("gates must contain strings.")
    if not isinstance(cfg["params_base"], dict):
        raise ValueError("params_base must be an object.")
    if not isinstance(cfg["grid"], dict):
        raise ValueError("grid must be an object.")

    for key in ["p_gate", "barrier", "tau"]:
        if key not in cfg["grid"]:
            raise ValueError(f"grid missing required key: {key}")
        vals = cfg["grid"][key]
        if not isinstance(vals, list) or len(vals) < 1:
            raise ValueError(f"grid['{key}'] must be a non-empty list.")
    if not all(isinstance(x, int) and x >= 0 for x in cfg["grid"]["tau"]):
        raise ValueError("grid['tau'] values must be integers >= 0.")


def _run_sweep(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gates = [str(g).lower() for g in cfg["gates"]]
    params_base = dict(cfg["params_base"])

    p_gate_vals = [float(x) for x in cfg["grid"]["p_gate"]]
    barrier_vals = [float(x) for x in cfg["grid"]["barrier"]]
    tau_vals = [int(x) for x in cfg["grid"]["tau"]]

    for gate_name in gates:
        for p_gate, barrier, tau in itertools.product(p_gate_vals, barrier_vals, tau_vals):
            params = dict(params_base)
            params["p_gate"] = p_gate
            params["barrier"] = barrier

            P, f_dict, meta = make_gate_lab(gate_name, params)
            n_micro = P.shape[0]
            mu_unif = np.ones(n_micro, dtype=float) / float(n_micro)

            err_gate_tau1 = gate_error_rate_kernel(
                P,
                f_dict["inputs"],
                f_dict["output"],
                meta["truth_table"],
                tau=1,
            )
            stabilities = [
                predicate_stability_kernel(P, f_dict[name], tau=1) for name in meta["stable_bits"]
            ]
            stability_min_tau1 = float(min(stabilities))

            rm_full = route_mismatch(P, f_dict["full"], tau=tau)
            comm_full = distribution_commutation_defect(mu_unif, P, f_dict["full"], tau=tau)

            if "parity" in f_dict:
                rm_parity = route_mismatch(P, f_dict["parity"], tau=tau)
                comm_parity = distribution_commutation_defect(mu_unif, P, f_dict["parity"], tau=tau)
            else:
                rm_parity = float("nan")
                comm_parity = float("nan")

            resolved = meta["params_resolved"]
            rows.append(
                {
                    "gate_name": gate_name,
                    "degeneracy": int(resolved["degeneracy"]),
                    "barrier": float(resolved["barrier"]),
                    "p_gate": float(resolved["p_gate"]),
                    "p_mem": float(resolved["p_mem"]),
                    "tau": int(tau),
                    "err_gate_tau1": float(err_gate_tau1),
                    "stability_min_tau1": float(stability_min_tau1),
                    "rm_full": float(rm_full),
                    "rm_parity": float(rm_parity),
                    "comm_full": float(comm_full),
                    "comm_parity": float(comm_parity),
                }
            )

    rows.sort(key=lambda r: (r["gate_name"], r["barrier"], r["p_gate"], r["tau"]))
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REQUIRED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _format_cell(row.get(k)) for k in REQUIRED_COLUMNS})


def _format_cell(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if math.isnan(x):
            return "nan"
        return format(x, ".12g")
    return str(value)


if __name__ == "__main__":
    main()
