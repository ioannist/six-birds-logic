"""Deterministically reproduce all current experiment outputs and write a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


EXPERIMENTS: list[tuple[str, list[str]]] = [
    (
        "exp_parity_vs_and",
        ["experiments/run_sweep.py", "--config", "configs/exp_parity_vs_and.json"],
    ),
    (
        "exp_parity_robustness",
        [
            "experiments/run_parity_robustness.py",
            "--config",
            "configs/exp_parity_robustness.json",
        ],
    ),
    (
        "exp_gate_phase_diagram",
        [
            "experiments/run_gate_phase_diagram.py",
            "--config",
            "configs/exp_gate_phase_diagram.json",
        ],
    ),
    (
        "exp_reversible_vs_erased",
        [
            "experiments/run_reversible_vs_erased.py",
            "--config",
            "configs/exp_reversible_vs_erased.json",
        ],
    ),
    ("exp_discovery_smoke", ["experiments/run_discovery_smoke.py"]),
    (
        "exp_gate_discovery",
        [
            "experiments/run_gate_discovery.py",
            "--config",
            "configs/exp_gate_discovery.json",
        ],
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce all experiment outputs.")
    parser.add_argument("--results-dir", default="results", help="Base results directory.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (repo_root / results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    for exp_name, _ in EXPERIMENTS:
        exp_dir = results_dir / exp_name
        if exp_dir.exists():
            shutil.rmtree(exp_dir)

    for exp_name, cmd_parts in EXPERIMENTS:
        cmd = [sys.executable, str(repo_root / cmd_parts[0]), *cmd_parts[1:], "--results-dir", str(results_dir)]
        subprocess.run(cmd, check=True, cwd=repo_root)
        print(f"DONE {exp_name}")

    manifest_path = results_dir / "final_manifest.json"
    payload = build_manifest(results_dir)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2)
        fh.write("\n")
    print(f"WROTE {display_path(manifest_path, repo_root)}")


def build_manifest(results_dir: Path) -> dict[str, object]:
    artifacts: list[dict[str, object]] = []
    for exp_name, _ in EXPERIMENTS:
        exp_dir = results_dir / exp_name
        for file_path in sorted(p for p in exp_dir.rglob("*") if p.is_file()):
            if file_path.name == "final_manifest.json":
                continue
            rel = file_path.relative_to(results_dir).as_posix()
            artifacts.append(
                {
                    "path": rel,
                    "sha256": sha256_file(file_path),
                    "size_bytes": file_path.stat().st_size,
                }
            )
    artifacts.sort(key=lambda item: str(item["path"]))
    return {
        "experiments": [name for name, _ in EXPERIMENTS],
        "total_artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
