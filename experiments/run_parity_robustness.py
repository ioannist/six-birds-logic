"""Parity robustness sweep: parity lens vs AND lens vs random balanced binary lenses."""

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
from emergent_logic.generator import make_gate_lab
from emergent_logic.metrics import induced_macro_kernel, route_mismatch


REQUIRED_COLUMNS = ["p_leak", "tau", "lens_type", "lens_id", "rm", "stability", "error"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parity robustness experiment.")
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config.")
    parser.add_argument("--results-dir", default="results", help="Base results directory.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_config(cfg_path)
    _validate_config(cfg)

    rows, stats = _run(cfg)

    out_dir = Path(args.results_dir) / str(cfg["exp_name"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.csv"
    _write_csv(summary_path, rows)

    with (out_dir / "config_used.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(cfg, fh, sort_keys=True, indent=2)
        fh.write("\n")

    with (out_dir / "stats.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(stats, fh, sort_keys=True, indent=2)
        fh.write("\n")

    print(f"win_rate={stats['win_rate']:.6f} (wins={stats['wins']} / total={stats['n_gridpoints']})")
    for key in ["p_leak=0.05,tau=1", "p_leak=0.2,tau=1"]:
        if key in stats["representative"]:
            rep = stats["representative"][key]
            print(
                f"{key}: rm_parity={rep['rm_parity']:.12g} "
                f"rm_median_random={rep['rm_median_random']:.12g}"
            )


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
        raise ValueError("Config root must be an object.")
    return loaded


def _validate_config(cfg: dict[str, Any]) -> None:
    for key in ["exp_name", "seed", "gate_name", "params_base", "grid", "n_random", "random_mode"]:
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")
    if str(cfg["gate_name"]).lower().replace("-", "_") != "parity_sector":
        raise ValueError("gate_name must be parity_sector for this experiment.")
    if not isinstance(cfg["seed"], int):
        raise ValueError("seed must be an integer.")
    if not isinstance(cfg["params_base"], dict):
        raise ValueError("params_base must be an object.")
    if not isinstance(cfg["grid"], dict):
        raise ValueError("grid must be an object.")
    for key in ["p_leak", "tau"]:
        if key not in cfg["grid"] or not isinstance(cfg["grid"][key], list) or len(cfg["grid"][key]) < 1:
            raise ValueError(f"grid.{key} must be a non-empty list.")
    if not all(isinstance(x, int) and x >= 0 for x in cfg["grid"]["tau"]):
        raise ValueError("grid.tau values must be integers >= 0.")
    if not isinstance(cfg["n_random"], int) or cfg["n_random"] < 1:
        raise ValueError("n_random must be an integer >= 1.")
    if str(cfg["random_mode"]) != "macro_balanced":
        raise ValueError("random_mode must be macro_balanced.")


def _run(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = int(cfg["seed"])
    n_random = int(cfg["n_random"])
    params_base = dict(cfg["params_base"])
    p_leak_vals = [float(x) for x in cfg["grid"]["p_leak"]]
    tau_vals = [int(x) for x in cfg["grid"]["tau"]]

    balanced = _balanced_binary_macro_assignments()

    rows: list[dict[str, Any]] = []
    grid_stats: dict[tuple[float, int], dict[str, float]] = {}

    for p_leak, tau in itertools.product(p_leak_vals, tau_vals):
        params = dict(params_base)
        params["p_leak"] = p_leak
        P, f_dict, _meta = make_gate_lab("parity_sector", params=params)
        n_micro = P.shape[0]
        mu_unif = np.ones(n_micro, dtype=float) / float(n_micro)

        f_par = f_dict["parity"].astype(int, copy=False)
        f_and = (f_dict["a"] & f_dict["b"]).astype(int, copy=False)
        f_full = f_dict["full"].astype(int, copy=False)

        rm_par = route_mismatch(P, f_par, tau=tau)
        rows.append(_row_for_lens(p_leak, tau, "parity", "", P, f_par, mu_unif))

        rows.append(_row_for_lens(p_leak, tau, "and", "", P, f_and, mu_unif))

        rng = np.random.default_rng(np.random.SeedSequence([seed, int(round(p_leak * 1_000_000)), tau]))
        random_rms: list[float] = []
        for lens_id in range(n_random):
            choice = int(rng.integers(0, balanced.shape[0]))
            macro_assign = balanced[choice]
            f_rand = macro_assign[f_full].astype(int, copy=False)
            row = _row_for_lens(p_leak, tau, "random", lens_id, P, f_rand, mu_unif)
            random_rms.append(float(row["rm"]))
            rows.append(row)

        grid_stats[(p_leak, tau)] = {
            "rm_parity": float(rm_par),
            "rm_median_random": float(np.median(np.asarray(random_rms, dtype=float))),
        }

    rows.sort(key=lambda r: (float(r["p_leak"]), int(r["tau"]), str(r["lens_type"]), str(r["lens_id"])))

    wins = 0
    for (_, _), vals in grid_stats.items():
        if vals["rm_parity"] < vals["rm_median_random"]:
            wins += 1
    n_grid = len(grid_stats)
    win_rate = float(wins) / float(n_grid) if n_grid > 0 else float("nan")

    rep: dict[str, dict[str, float]] = {}
    for p_ref in [0.05, 0.2]:
        key_tup = _find_grid_key(grid_stats, p_ref, 1)
        if key_tup is None:
            continue
        vals = grid_stats[key_tup]
        rep[_rep_key(p_ref, 1)] = {
            "rm_parity": float(vals["rm_parity"]),
            "rm_median_random": float(vals["rm_median_random"]),
        }

    stats = {
        "win_rate": win_rate,
        "wins": wins,
        "n_gridpoints": n_grid,
        "representative": rep,
    }
    return rows, stats


def _row_for_lens(
    p_leak: float,
    tau: int,
    lens_type: str,
    lens_id: int | str,
    P: np.ndarray,
    f_lens: np.ndarray,
    mu_unif: np.ndarray,
) -> dict[str, Any]:
    rm = float(route_mismatch(P, f_lens, tau=tau))
    stability = float(predicate_stability_kernel(P, pred=f_lens, tau=tau, mu=mu_unif))
    k = induced_macro_kernel(P, f_lens, tau=tau)
    counts = np.bincount(f_lens.astype(int, copy=False), minlength=2)
    nonempty = [label for label in [0, 1] if counts[label] > 0]
    if len(nonempty) == 0:
        err = float("nan")
    else:
        errs = [1.0 - float(np.max(k[label, :])) for label in nonempty]
        err = float(np.mean(np.asarray(errs, dtype=float)))
    return {
        "p_leak": float(p_leak),
        "tau": int(tau),
        "lens_type": lens_type,
        "lens_id": lens_id,
        "rm": rm,
        "stability": stability,
        "error": err,
    }


def _balanced_binary_macro_assignments() -> np.ndarray:
    rows: list[np.ndarray] = []
    for ones in itertools.combinations(range(4), 2):
        vec = np.zeros(4, dtype=int)
        vec[list(ones)] = 1
        rows.append(vec)
    return np.asarray(rows, dtype=int)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REQUIRED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in REQUIRED_COLUMNS})


def _fmt(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        x = float(v)
        if math.isnan(x):
            return "nan"
        return format(x, ".12g")
    return str(v)


def _find_grid_key(
    grid_stats: dict[tuple[float, int], dict[str, float]],
    p_target: float,
    tau_target: int,
) -> tuple[float, int] | None:
    for p, tau in grid_stats:
        if tau == tau_target and abs(p - p_target) < 1e-12:
            return (p, tau)
    return None


def _rep_key(p_leak: float, tau: int) -> str:
    p_txt = format(float(p_leak), "g")
    return f"p_leak={p_txt},tau={int(tau)}"


if __name__ == "__main__":
    main()
