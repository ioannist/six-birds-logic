"""Gate closure phase diagram runner for NOT/CNOT/AND labs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from emergent_logic.generator import make_gate_lab
from emergent_logic.markov import power, stationary_distribution
from emergent_logic.metrics import route_mismatch


REQUIRED_COLUMNS = [
    "row_id",
    "gate_name",
    "degeneracy",
    "barrier",
    "p_gate",
    "p_mem",
    "tau",
    "err_truth",
    "err_induced",
    "H_out_given_in",
    "rm_output",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gate closure phase-diagram sweep.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config.")
    parser.add_argument("--results-dir", default="results", help="Base results directory.")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    _validate_config(cfg)

    rows = _run(cfg)
    rows.sort(key=lambda r: (str(r["gate_name"]), float(r["barrier"]), float(r["p_gate"]), int(r["tau"])))
    for idx, row in enumerate(rows):
        row["row_id"] = idx

    stats = _build_stats(rows)

    out_dir = Path(args.results_dir) / str(cfg["exp_name"])
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(out_dir / "summary.csv", rows)
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
    for key in ["exp_name", "seed", "gates", "params_base", "grid"]:
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")
    if not isinstance(cfg["exp_name"], str) or not cfg["exp_name"]:
        raise ValueError("exp_name must be a non-empty string.")
    if not isinstance(cfg["seed"], int):
        raise ValueError("seed must be an integer.")
    if not isinstance(cfg["gates"], list) or not cfg["gates"]:
        raise ValueError("gates must be a non-empty list.")
    if not all(isinstance(g, str) for g in cfg["gates"]):
        raise ValueError("gates entries must be strings.")
    if not isinstance(cfg["params_base"], dict):
        raise ValueError("params_base must be an object.")
    if not isinstance(cfg["grid"], dict):
        raise ValueError("grid must be an object.")
    for key in ["p_gate", "barrier", "tau"]:
        vals = cfg["grid"].get(key)
        if not isinstance(vals, list) or len(vals) == 0:
            raise ValueError(f"grid.{key} must be a non-empty list.")
    if not all(isinstance(t, int) and t >= 0 for t in cfg["grid"]["tau"]):
        raise ValueError("grid.tau values must be integers >= 0.")


def _run(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gates = [str(g).lower() for g in cfg["gates"]]
    params_base = dict(cfg["params_base"])
    p_gate_vals = [float(x) for x in cfg["grid"]["p_gate"]]
    barrier_vals = [float(x) for x in cfg["grid"]["barrier"]]
    tau_vals = [int(x) for x in cfg["grid"]["tau"]]

    for gate_name in gates:
        for barrier, p_gate, tau in itertools.product(barrier_vals, p_gate_vals, tau_vals):
            params = dict(params_base)
            params["barrier"] = barrier
            params["p_gate"] = p_gate

            P, f_dict, meta = make_gate_lab(gate_name, params)
            P_tau = power(P, tau)

            f_in = f_dict["inputs"].astype(int, copy=False)
            f_out = f_dict["output"].astype(int, copy=False)
            tt = np.asarray(meta["truth_table"], dtype=int)

            err_truth, err_induced, h_out_given_in = _channel_metrics(P_tau, f_in, f_out, tt)
            pi = stationary_distribution(P)
            rm_output = float(route_mismatch(P, f_out, tau=tau, weights=pi))

            resolved = meta["params_resolved"]
            rows.append(
                {
                    "gate_name": gate_name,
                    "degeneracy": int(resolved["degeneracy"]),
                    "barrier": float(barrier),
                    "p_gate": float(p_gate),
                    "p_mem": float(resolved["p_mem"]),
                    "tau": int(tau),
                    "err_truth": float(err_truth),
                    "err_induced": float(err_induced),
                    "H_out_given_in": float(h_out_given_in),
                    "rm_output": float(rm_output),
                }
            )
    return rows


def _channel_metrics(
    P_tau: np.ndarray,
    f_in: np.ndarray,
    f_out: np.ndarray,
    truth_table: np.ndarray,
) -> tuple[float, float, float]:
    n_micro = P_tau.shape[0]
    if f_in.shape[0] != n_micro or f_out.shape[0] != n_micro:
        raise ValueError("f_in and f_out must match kernel size.")
    n_inputs = int(truth_table.shape[0])
    if np.any(f_in < 0) or np.any(f_in >= n_inputs):
        raise ValueError("f_in has labels outside truth table range.")

    out1_mask = (f_out == 1).astype(float)
    p1 = np.zeros(n_inputs, dtype=float)

    for u in range(n_inputs):
        idx = np.where(f_in == u)[0]
        if idx.size == 0:
            raise ValueError(f"Input pattern {u} has no supporting microstates.")
        mu = np.zeros(n_micro, dtype=float)
        mu[idx] = 1.0 / float(idx.size)
        dist_u = mu @ P_tau
        p1[u] = float(np.dot(dist_u, out1_mask))

    err_u = np.where(truth_table == 1, 1.0 - p1, p1)
    err_truth = float(np.mean(err_u))

    err_induced = float(np.mean(np.minimum(p1, 1.0 - p1)))

    h_u = _binary_entropy(p1)
    h_out_given_in = float(np.mean(h_u))
    return err_truth, err_induced, h_out_given_in


def _binary_entropy(p1: np.ndarray) -> np.ndarray:
    p = np.asarray(p1, dtype=float)
    q = 1.0 - p
    out = np.zeros_like(p, dtype=float)
    mask_p = p > 0.0
    mask_q = q > 0.0
    out[mask_p] -= p[mask_p] * np.log2(p[mask_p])
    out[mask_q] -= q[mask_q] * np.log2(q[mask_q])
    return out


def _build_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_rows = len(rows)
    per_gate: dict[str, dict[str, float]] = {}
    for gate_name in sorted({str(r["gate_name"]) for r in rows}):
        gate_rows = [r for r in rows if str(r["gate_name"]) == gate_name]
        e0 = [float(r["err_truth"]) for r in gate_rows if abs(float(r["p_gate"]) - 0.0) < 1e-12]
        e1 = [float(r["err_truth"]) for r in gate_rows if abs(float(r["p_gate"]) - 0.1) < 1e-12]
        i0 = [float(r["err_induced"]) for r in gate_rows if abs(float(r["p_gate"]) - 0.0) < 1e-12]
        i1 = [float(r["err_induced"]) for r in gate_rows if abs(float(r["p_gate"]) - 0.1) < 1e-12]
        per_gate[gate_name] = {
            "mean_err_truth_pgate_0": float(np.mean(np.asarray(e0, dtype=float))),
            "mean_err_truth_pgate_0.1": float(np.mean(np.asarray(e1, dtype=float))),
            "mean_err_induced_pgate_0": float(np.mean(np.asarray(i0, dtype=float))),
            "mean_err_induced_pgate_0.1": float(np.mean(np.asarray(i1, dtype=float))),
        }

    anomalies: list[dict[str, Any]] = []
    metrics = ["err_truth", "err_induced", "H_out_given_in"]
    groups: dict[tuple[str, float, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["gate_name"]), float(row["barrier"]), int(row["tau"]))
        groups.setdefault(key, []).append(row)

    for (gate_name, barrier, tau), group_rows in groups.items():
        ordered = sorted(group_rows, key=lambda r: float(r["p_gate"]))
        for metric in metrics:
            for low, high in zip(ordered[:-1], ordered[1:]):
                v_low = float(low[metric])
                v_high = float(high[metric])
                if v_high < v_low - 1e-12:
                    anomalies.append(
                        {
                            "metric": metric,
                            "gate_name": gate_name,
                            "barrier": barrier,
                            "tau": tau,
                            "p_gate_low": float(low["p_gate"]),
                            "p_gate_high": float(high["p_gate"]),
                            "row_id_low": int(low["row_id"]),
                            "row_id_high": int(high["row_id"]),
                            "value_low": v_low,
                            "value_high": v_high,
                        }
                    )

    return {"n_rows": n_rows, "per_gate": per_gate, "anomalies": anomalies}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REQUIRED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _fmt_cell(row[col]) for col in REQUIRED_COLUMNS})


def _fmt_cell(value: Any) -> str:
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
