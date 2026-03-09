"""Validate sealed repository state from frozen results and hygiene checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final sealed repository state.")
    parser.add_argument("--results-dir", default="results", help="Results directory.")
    parser.add_argument("--repo-root", default=".", help="Repository root directory.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (repo_root / results_dir).resolve()

    claims_dir = results_dir / "final_claims"
    manifest_path = results_dir / "final_manifest.json"
    claims_path = claims_dir / "claims.json"
    report_path = claims_dir / "seal_report.json"

    checks: dict[str, dict[str, Any]] = {}
    record_check(checks, "manifest_exists", manifest_path.is_file(), display_path(manifest_path, repo_root))
    record_check(checks, "claims_exists", claims_path.is_file(), display_path(claims_path, repo_root))

    claims: dict[str, Any] = {}
    if claims_path.is_file():
        claims = load_json_object(claims_path)
    if manifest_path.is_file():
        _ = load_json_object(manifest_path)

    check_parity_claims(checks, claims)
    check_phase_claims(checks, claims)
    check_reversible_claims(checks, claims)
    check_discovery_claims(checks, claims)
    check_gate_discovery_claims(checks, claims)
    check_repo_hygiene(checks, repo_root)

    overall_ok = all(entry.get("ok", False) for entry in checks.values())
    report = {"ok": overall_ok, "checks": checks}

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, sort_keys=True, indent=2)
        fh.write("\n")

    raise SystemExit(0 if overall_ok else 1)


def check_parity_claims(checks: dict[str, dict[str, Any]], claims: dict[str, Any]) -> None:
    win = float_or_none(claims.get("parity_win_rate"))
    p005 = float_or_none(claims.get("parity_rep_p005_tau1_rm"))
    p005_rand = float_or_none(claims.get("random_rep_p005_tau1_median_rm"))
    p020 = float_or_none(claims.get("parity_rep_p020_tau1_rm"))
    p020_rand = float_or_none(claims.get("random_rep_p020_tau1_median_rm"))

    record_check(checks, "parity_win_rate_threshold", is_valid_number(win) and win >= 0.80, win)
    record_check(
        checks,
        "parity_rep_p005_beats_random",
        is_valid_number(p005) and is_valid_number(p005_rand) and p005 < p005_rand,
        {"parity": p005, "random_median": p005_rand},
    )
    record_check(
        checks,
        "parity_rep_p020_beats_random",
        is_valid_number(p020) and is_valid_number(p020_rand) and p020 < p020_rand,
        {"parity": p020, "random_median": p020_rand},
    )


def check_phase_claims(checks: dict[str, dict[str, Any]], claims: dict[str, Any]) -> None:
    n_rows = int_or_none(claims.get("phase_n_rows"))
    anomaly_count = int_or_none(claims.get("phase_anomaly_count"))
    not_p0 = float_or_none(claims.get("phase_not_err_truth_p0"))
    not_p01 = float_or_none(claims.get("phase_not_err_truth_p01"))
    cnot_p0 = float_or_none(claims.get("phase_cnot_err_truth_p0"))
    cnot_p01 = float_or_none(claims.get("phase_cnot_err_truth_p01"))
    and_p0 = float_or_none(claims.get("phase_and_err_truth_p0"))
    and_p01 = float_or_none(claims.get("phase_and_err_truth_p01"))

    record_check(checks, "phase_row_count", n_rows == 108, n_rows)
    record_check(checks, "phase_no_anomalies", anomaly_count == 0, anomaly_count)
    record_check(
        checks,
        "phase_not_monotone",
        is_valid_number(not_p01) and is_valid_number(not_p0) and not_p01 >= not_p0 - 1e-12,
        {"p0": not_p0, "p01": not_p01},
    )
    record_check(
        checks,
        "phase_cnot_monotone",
        is_valid_number(cnot_p01) and is_valid_number(cnot_p0) and cnot_p01 >= cnot_p0 - 1e-12,
        {"p0": cnot_p0, "p01": cnot_p01},
    )
    record_check(
        checks,
        "phase_and_monotone",
        is_valid_number(and_p01) and is_valid_number(and_p0) and and_p01 >= and_p0 - 1e-12,
        {"p0": and_p0, "p01": and_p01},
    )


def check_reversible_claims(checks: dict[str, dict[str, Any]], claims: dict[str, Any]) -> None:
    ratio = float_or_none(claims.get("rev_vs_erased_ratio_unretained_input_info"))
    closure_delta = float_or_none(claims.get("rev_vs_erased_closure_defect_delta"))
    rm_delta = float_or_none(claims.get("rev_vs_erased_rm_delta"))

    record_check(
        checks,
        "rev_erased_ratio_threshold",
        is_valid_number(ratio) and ratio > 5.0,
        ratio,
    )
    record_check(
        checks,
        "rev_erased_closure_delta",
        is_valid_number(closure_delta) and closure_delta > 0.5,
        closure_delta,
    )
    record_check(
        checks,
        "rev_erased_rm_delta",
        is_valid_number(rm_delta) and rm_delta > 0.5,
        rm_delta,
    )


def check_discovery_claims(checks: dict[str, dict[str, Any]], claims: dict[str, Any]) -> None:
    not_agreement = float_or_none(claims.get("discovery_not_best_agreement"))
    parity_agreement = float_or_none(claims.get("discovery_parity_best_agreement"))

    record_check(
        checks,
        "discovery_not_agreement",
        is_valid_number(not_agreement) and not_agreement >= 0.95,
        not_agreement,
    )
    record_check(
        checks,
        "discovery_parity_agreement",
        is_valid_number(parity_agreement) and parity_agreement >= 0.95,
        parity_agreement,
    )


def check_gate_discovery_claims(checks: dict[str, dict[str, Any]], claims: dict[str, Any]) -> None:
    table = claims.get("not_gate_truth_table_matrix_flat")
    error = float_or_none(claims.get("not_gate_error"))

    record_check(checks, "not_gate_truth_table", table == [0, 1, 1, 0], table)
    record_check(
        checks,
        "not_gate_error_threshold",
        is_valid_number(error) and error < 0.10,
        error,
    )


def check_repo_hygiene(checks: dict[str, dict[str, Any]], repo_root: Path) -> None:
    build_dir = repo_root / "build"
    record_check(checks, "repo_no_build_dir", not build_dir.is_dir(), build_dir.is_dir())

    egg_info_dirs = sorted(
        display_path(path, repo_root)
        for path in repo_root.rglob("*.egg-info")
        if path.is_dir()
    )
    record_check(checks, "repo_no_egg_info", len(egg_info_dirs) == 0, egg_info_dirs)

    stray_paths = [
        repo_root / "lean" / "Main.lean",
        repo_root / "lean" / "LogicClosure.lean",
        repo_root / "lean" / "lakefile.toml",
        repo_root / "lean" / "README.md",
        repo_root / "lean" / "lean-toolchain",
        repo_root / "lean" / ".gitignore",
        repo_root / "lean" / ".github",
    ]
    present = [display_path(path, repo_root) for path in stray_paths if path.exists()]
    record_check(checks, "repo_no_stray_outer_lean_scaffold", len(present) == 0, present)


def record_check(
    checks: dict[str, dict[str, Any]],
    name: str,
    ok: bool,
    value: Any,
) -> None:
    checks[name] = {"ok": bool(ok), "value": value}


def load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_valid_number(value: float | None) -> bool:
    return value is not None and value == value and value not in (float("inf"), float("-inf"))


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
