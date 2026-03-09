"""Freeze headline claims and figure-ready tables from existing result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_INPUTS = [
    "exp_parity_robustness/summary.csv",
    "exp_parity_robustness/stats.json",
    "exp_gate_phase_diagram/summary.csv",
    "exp_gate_phase_diagram/stats.json",
    "exp_reversible_vs_erased/comparison.csv",
    "exp_reversible_vs_erased/stats.json",
    "exp_discovery_smoke/summary.json",
    "exp_discovery_smoke/not_candidates.csv",
    "exp_discovery_smoke/parity_sector_candidates.csv",
    "exp_gate_discovery/gates.json",
    "exp_gate_discovery/not_output_candidates.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze claims bundle from existing results.")
    parser.add_argument("--results-dir", default="results", help="Base results directory.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (repo_root / results_dir).resolve()

    ensure_required_inputs(results_dir)

    parity_summary = read_csv_rows(results_dir / "exp_parity_robustness" / "summary.csv")
    parity_stats = read_json(results_dir / "exp_parity_robustness" / "stats.json")

    phase_summary = read_csv_rows(results_dir / "exp_gate_phase_diagram" / "summary.csv")
    phase_stats = read_json(results_dir / "exp_gate_phase_diagram" / "stats.json")

    rev_comp = read_csv_rows(results_dir / "exp_reversible_vs_erased" / "comparison.csv")
    rev_stats = read_json(results_dir / "exp_reversible_vs_erased" / "stats.json")

    discovery_summary = read_json(results_dir / "exp_discovery_smoke" / "summary.json")
    disc_not_rows = read_csv_rows(results_dir / "exp_discovery_smoke" / "not_candidates.csv")
    disc_par_rows = read_csv_rows(results_dir / "exp_discovery_smoke" / "parity_sector_candidates.csv")

    gates = read_json(results_dir / "exp_gate_discovery" / "gates.json")

    out_dir = results_dir / "final_claims"
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_parity_rows = build_figure_parity_robustness(parity_summary, parity_stats)
    write_csv(
        out_dir / "figure_parity_robustness.csv",
        [
            "p_leak",
            "tau",
            "parity_rm",
            "and_rm",
            "median_random_rm",
            "parity_stability",
            "and_stability",
            "parity_error",
            "and_error",
            "win_vs_random_median",
        ],
        figure_parity_rows,
    )

    figure_phase_rows = build_figure_gate_phase(phase_summary)
    write_csv(
        out_dir / "figure_gate_phase.csv",
        [
            "row_id",
            "gate_name",
            "barrier",
            "p_gate",
            "tau",
            "err_truth",
            "err_induced",
            "H_out_given_in",
            "rm_output",
        ],
        figure_phase_rows,
    )

    table_rev_rows = build_reversible_table(rev_comp)
    write_csv(
        out_dir / "table_reversible_vs_erased.csv",
        [
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
        ],
        table_rev_rows,
    )

    table_disc_rows = build_partition_discovery_table(
        discovery_summary,
        disc_not_rows,
        disc_par_rows,
    )
    write_csv(
        out_dir / "table_partition_discovery.csv",
        [
            "lab_name",
            "best_agreement",
            "n_candidates",
            "best_score",
            "best_metastability",
            "best_rm",
            "top2_score",
            "top3_score",
        ],
        table_disc_rows,
    )

    table_gate_rows = build_gate_discovery_table(gates)
    write_csv(
        out_dir / "table_gate_discovery.csv",
        [
            "gate_name",
            "truth_table_bits",
            "truth_table_matrix_flat",
            "error",
            "entropy",
            "sample_size",
            "input_partition_score",
            "output_delta_I",
            "output_future_I",
            "output_current_I",
        ],
        table_gate_rows,
    )

    claims = build_claims(
        parity_stats=parity_stats,
        phase_stats=phase_stats,
        rev_stats=rev_stats,
        discovery_summary=discovery_summary,
        gates=gates,
    )
    with (out_dir / "claims.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(claims, fh, sort_keys=True, indent=2)
        fh.write("\n")


def ensure_required_inputs(results_dir: Path) -> None:
    missing = [rel for rel in REQUIRED_INPUTS if not (results_dir / rel).is_file()]
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required source files under {results_dir}:\n{lines}")


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return data


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: format_value(row.get(k, "")) for k in columns})


def build_figure_parity_robustness(
    summary_rows: list[dict[str, str]],
    stats: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, int], list[dict[str, str]]] = {}
    for row in summary_rows:
        key = (float(row["p_leak"]), int(row["tau"]))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    wins = 0
    for p_leak, tau in sorted(grouped.keys()):
        rows = grouped[(p_leak, tau)]
        parity = one_row(rows, "parity")
        and_row = one_row(rows, "and")
        random_rows = [r for r in rows if r["lens_type"] == "random"]
        random_rm = sorted(float(r["rm"]) for r in random_rows)
        median_random_rm = median(random_rm)
        parity_rm = float(parity["rm"])
        win = parity_rm < median_random_rm
        wins += int(win)
        out.append(
            {
                "p_leak": p_leak,
                "tau": tau,
                "parity_rm": parity_rm,
                "and_rm": float(and_row["rm"]),
                "median_random_rm": median_random_rm,
                "parity_stability": float(parity["stability"]),
                "and_stability": float(and_row["stability"]),
                "parity_error": float(parity["error"]),
                "and_error": float(and_row["error"]),
                "win_vs_random_median": int(win),
            }
        )

    if len(out) != 12:
        raise ValueError(f"Expected 12 parity-robustness rows, got {len(out)}.")

    computed_win_rate = wins / len(out)
    expected_win_rate = float(stats["win_rate"])
    if abs(computed_win_rate - expected_win_rate) > 1e-12:
        raise ValueError(
            f"Win-rate mismatch: computed {computed_win_rate} vs stats {expected_win_rate}."
        )

    return out


def one_row(rows: list[dict[str, str]], lens_type: str) -> dict[str, str]:
    matches = [row for row in rows if row["lens_type"] == lens_type]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row for lens_type={lens_type}, got {len(matches)}.")
    return matches[0]


def build_figure_gate_phase(summary_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    columns = [
        "row_id",
        "gate_name",
        "barrier",
        "p_gate",
        "tau",
        "err_truth",
        "err_induced",
        "H_out_given_in",
        "rm_output",
    ]
    out = [{col: row[col] for col in columns} for row in summary_rows]
    if len(out) != 108:
        raise ValueError(f"Expected 108 gate-phase rows, got {len(out)}.")
    return out


def build_reversible_table(comparison_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    columns = [
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
    out = [{col: row[col] for col in columns} for row in comparison_rows]
    if len(out) != 3:
        raise ValueError(f"Expected 3 reversible-vs-erased rows, got {len(out)}.")
    return out


def build_partition_discovery_table(
    summary: dict[str, Any],
    not_rows: list[dict[str, str]],
    parity_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows = [
        build_partition_row("not", summary, not_rows),
        build_partition_row("parity_sector", summary, parity_rows),
    ]
    rows.sort(key=lambda row: str(row["lab_name"]))
    if len(rows) != 2:
        raise ValueError(f"Expected 2 partition-discovery rows, got {len(rows)}.")
    return rows


def build_partition_row(
    lab_name: str,
    summary: dict[str, Any],
    candidate_rows: list[dict[str, str]],
) -> dict[str, Any]:
    if not candidate_rows:
        raise ValueError(f"No candidate rows for lab {lab_name}.")
    best = candidate_rows[0]
    top3 = summary.get(f"{lab_name}_top3_scores", [])
    top2_score = (
        float(top3[1])
        if len(top3) > 1
        else (float(candidate_rows[1]["score"]) if len(candidate_rows) > 1 else math.nan)
    )
    top3_score = (
        float(top3[2])
        if len(top3) > 2
        else (float(candidate_rows[2]["score"]) if len(candidate_rows) > 2 else math.nan)
    )
    return {
        "lab_name": lab_name,
        "best_agreement": float(summary[f"{lab_name}_best_agreement"]),
        "n_candidates": int(summary[f"{lab_name}_n_candidates"]),
        "best_score": float(best["score"]),
        "best_metastability": float(best["metastability"]),
        "best_rm": float(best["rm"]),
        "top2_score": top2_score,
        "top3_score": top3_score,
    }


def build_gate_discovery_table(gates: dict[str, Any]) -> list[dict[str, Any]]:
    not_gate = gates["not"]
    row = {
        "gate_name": "not",
        "truth_table_bits": json.dumps(not_gate["truth_table_bits"], separators=(",", ":")),
        "truth_table_matrix_flat": json.dumps(
            not_gate["truth_table_matrix_flat"], separators=(",", ":")
        ),
        "error": float(not_gate["error"]),
        "entropy": float(not_gate["entropy"]),
        "sample_size": int(not_gate["sample_size"]),
        "input_partition_score": float(not_gate["input_partition_score"]),
        "output_delta_I": float(not_gate["output_delta_I"]),
        "output_future_I": float(not_gate["output_future_I"]),
        "output_current_I": float(not_gate["output_current_I"]),
    }
    return [row]


def build_claims(
    parity_stats: dict[str, Any],
    phase_stats: dict[str, Any],
    rev_stats: dict[str, Any],
    discovery_summary: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    rep_005 = parity_stats["representative"]["p_leak=0.05,tau=1"]
    rep_020 = parity_stats["representative"]["p_leak=0.2,tau=1"]
    per_gate = phase_stats["per_gate"]
    not_gate = gates["not"]

    return {
        "parity_win_rate": float(parity_stats["win_rate"]),
        "parity_rep_p005_tau1_rm": float(rep_005["rm_parity"]),
        "random_rep_p005_tau1_median_rm": float(rep_005["rm_median_random"]),
        "parity_rep_p020_tau1_rm": float(rep_020["rm_parity"]),
        "random_rep_p020_tau1_median_rm": float(rep_020["rm_median_random"]),
        "phase_n_rows": int(phase_stats["n_rows"]),
        "phase_anomaly_count": int(len(phase_stats.get("anomalies", []))),
        "phase_not_err_truth_p0": float(per_gate["not"]["mean_err_truth_pgate_0"]),
        "phase_not_err_truth_p01": float(per_gate["not"]["mean_err_truth_pgate_0.1"]),
        "phase_cnot_err_truth_p0": float(per_gate["cnot"]["mean_err_truth_pgate_0"]),
        "phase_cnot_err_truth_p01": float(per_gate["cnot"]["mean_err_truth_pgate_0.1"]),
        "phase_and_err_truth_p0": float(per_gate["and"]["mean_err_truth_pgate_0"]),
        "phase_and_err_truth_p01": float(per_gate["and"]["mean_err_truth_pgate_0.1"]),
        "rev_vs_erased_ratio_unretained_input_info": float(
            rev_stats["ratio_unretained_input_info"]
        ),
        "rev_vs_erased_closure_defect_delta": float(rev_stats["closure_defect_delta"]),
        "rev_vs_erased_rm_delta": float(rev_stats["rm_delta"]),
        "discovery_not_best_agreement": float(discovery_summary["not_best_agreement"]),
        "discovery_parity_best_agreement": float(
            discovery_summary["parity_sector_best_agreement"]
        ),
        "not_gate_truth_table_matrix_flat": [
            int(x) for x in not_gate["truth_table_matrix_flat"]
        ],
        "not_gate_error": float(not_gate["error"]),
        "not_gate_entropy": float(not_gate["entropy"]),
        "not_gate_sample_size": int(not_gate["sample_size"]),
    }


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute median of empty list.")
    n = len(values)
    mid = n // 2
    if n % 2 == 1:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def format_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return format(value, ".12g")
    return str(value)


if __name__ == "__main__":
    main()
