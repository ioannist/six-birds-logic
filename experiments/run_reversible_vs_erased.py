"""Compare reversible embedding and erased-output views on the same CNOT lab."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from emergent_logic.accounting import (
    apparent_entropy_production_rate,
    channel_from_kernel,
    channel_information_measures,
    entropy_production_rate,
)
from emergent_logic.generator import make_gate_lab
from emergent_logic.metrics import distribution_commutation_defect, route_mismatch
from emergent_logic.markov import stationary_distribution


COLUMNS = [
    "view",
    "n_states_view",
    "closure_defect",
    "rm_view",
    "epr_view",
    "H_in",
    "H_out",
    "H_out_given_in",
    "I_in_out",
    "entropy_drop",
    "unretained_input_info",
]

VIEW_ORDER = ["cnot_micro", "cnot_macro", "xor_erased_macro"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reversible-vs-erased CNOT comparison.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config.")
    parser.add_argument("--results-dir", default="results", help="Base results directory.")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    _validate_config(cfg)

    rows, stats = _run(cfg)

    out_dir = Path(args.results_dir) / str(cfg["exp_name"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row[k]) for k in COLUMNS})

    with (out_dir / "config_used.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(cfg, fh, sort_keys=True, indent=2)
        fh.write("\n")

    with (out_dir / "stats.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(stats, fh, sort_keys=True, indent=2)
        fh.write("\n")


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
    for key in ["exp_name", "gate_name", "params", "tau", "probe_macro_input"]:
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")

    gate_name = str(cfg["gate_name"]).lower().replace("-", "_")
    if gate_name != "cnot":
        raise ValueError("gate_name must be 'cnot' for this comparison.")

    if not isinstance(cfg["params"], dict):
        raise ValueError("params must be an object.")
    if not isinstance(cfg["tau"], int) or int(cfg["tau"]) < 0:
        raise ValueError("tau must be an integer >= 0.")
    if not isinstance(cfg["probe_macro_input"], int) or int(cfg["probe_macro_input"]) < 0:
        raise ValueError("probe_macro_input must be a nonnegative integer.")


def _run(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    tau = int(cfg["tau"])
    probe_macro_input = int(cfg["probe_macro_input"])

    P, f_dict, _meta = make_gate_lab("cnot", dict(cfg["params"]))
    n_micro = P.shape[0]
    f_full = f_dict["full"].astype(int, copy=False)
    f_inputs = f_dict["inputs"].astype(int, copy=False)
    f_output = f_dict["output"].astype(int, copy=False)
    f_id = np.arange(n_micro, dtype=int)

    idx_probe_candidates = np.where(f_full == probe_macro_input)[0]
    if idx_probe_candidates.size == 0:
        raise ValueError("probe_macro_input has no supporting microstate.")
    idx_probe = int(idx_probe_candidates[0])
    mu_probe = np.zeros(n_micro, dtype=float)
    mu_probe[idx_probe] = 1.0

    pi = stationary_distribution(P)

    rows = []

    # View 1: cnot_micro
    channel_micro = channel_from_kernel(P, f_inputs, f_full, tau=tau)
    info_micro = channel_information_measures(channel_micro)
    rows.append(
        {
            "view": "cnot_micro",
            "n_states_view": int(n_micro),
            "closure_defect": float(distribution_commutation_defect(mu_probe, P, f_id, tau=tau)),
            "rm_view": float(route_mismatch(P, f_id, tau=tau, weights=pi)),
            "epr_view": float(entropy_production_rate(P)),
            "H_in": float(info_micro.H_in),
            "H_out": float(info_micro.H_out),
            "H_out_given_in": float(info_micro.H_out_given_in),
            "I_in_out": float(info_micro.I_in_out),
            "entropy_drop": float(info_micro.entropy_drop),
            "unretained_input_info": float(info_micro.unretained_input_info),
        }
    )

    # View 2: cnot_macro
    channel_macro = channel_from_kernel(P, f_inputs, f_full, tau=tau)
    info_macro = channel_information_measures(channel_macro)
    rows.append(
        {
            "view": "cnot_macro",
            "n_states_view": int(f_full.max()) + 1,
            "closure_defect": float(distribution_commutation_defect(mu_probe, P, f_full, tau=tau)),
            "rm_view": float(route_mismatch(P, f_full, tau=tau, weights=pi)),
            "epr_view": float(apparent_entropy_production_rate(P, f_full, tau=tau)),
            "H_in": float(info_macro.H_in),
            "H_out": float(info_macro.H_out),
            "H_out_given_in": float(info_macro.H_out_given_in),
            "I_in_out": float(info_macro.I_in_out),
            "entropy_drop": float(info_macro.entropy_drop),
            "unretained_input_info": float(info_macro.unretained_input_info),
        }
    )

    # View 3: xor_erased_macro
    channel_xor = channel_from_kernel(P, f_inputs, f_output, tau=tau)
    info_xor = channel_information_measures(channel_xor)
    rows.append(
        {
            "view": "xor_erased_macro",
            "n_states_view": int(f_output.max()) + 1,
            "closure_defect": float(distribution_commutation_defect(mu_probe, P, f_output, tau=tau)),
            "rm_view": float(route_mismatch(P, f_output, tau=tau, weights=pi)),
            "epr_view": float(apparent_entropy_production_rate(P, f_output, tau=tau)),
            "H_in": float(info_xor.H_in),
            "H_out": float(info_xor.H_out),
            "H_out_given_in": float(info_xor.H_out_given_in),
            "I_in_out": float(info_xor.I_in_out),
            "entropy_drop": float(info_xor.entropy_drop),
            "unretained_input_info": float(info_xor.unretained_input_info),
        }
    )

    rows.sort(key=lambda r: VIEW_ORDER.index(str(r["view"])))

    by_view = {str(r["view"]): r for r in rows}
    denom = float(by_view["cnot_macro"]["unretained_input_info"])
    num = float(by_view["xor_erased_macro"]["unretained_input_info"])
    ratio = float(np.inf) if abs(denom) < 1e-15 else float(num / denom)

    stats = {
        "ratio_unretained_input_info": ratio,
        "cnot_macro_unretained_input_info": float(by_view["cnot_macro"]["unretained_input_info"]),
        "xor_erased_macro_unretained_input_info": float(by_view["xor_erased_macro"]["unretained_input_info"]),
        "closure_defect_delta": float(
            by_view["xor_erased_macro"]["closure_defect"] - by_view["cnot_macro"]["closure_defect"]
        ),
        "rm_delta": float(by_view["xor_erased_macro"]["rm_view"] - by_view["cnot_macro"]["rm_view"]),
    }

    return rows, stats


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
