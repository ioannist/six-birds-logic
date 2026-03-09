"""Smoke experiment for spectral binary partition discovery."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from emergent_logic.discovery import partition_agreement, spectral_binary_candidates
from emergent_logic.generator import make_gate_lab


COLUMNS = [
    "rank",
    "lab_name",
    "threshold",
    "eigenvalue2",
    "score",
    "metastability",
    "rm",
    "size0",
    "size1",
    "agreement_to_truth",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run discovery smoke experiment.")
    parser.add_argument("--results-dir", default="results", help="Base results directory.")
    args = parser.parse_args()

    out_dir = Path(args.results_dir) / "exp_discovery_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}

    labs = [
        (
            "not",
            "not",
            {"degeneracy": 3, "barrier": 8.0, "base_mem_noise": 0.05, "p_gate": 0.01},
            "a",
        ),
        (
            "parity_sector",
            "parity_sector",
            {"degeneracy": 3, "p_leak": 0.05},
            "parity",
        ),
    ]

    for gate_name, lab_name, params, truth_key in labs:
        P, f_dict, _meta = make_gate_lab(gate_name, params)
        truth = f_dict[truth_key]
        candidates = spectral_binary_candidates(P, tau=1, max_thresholds=None)

        csv_path = out_dir / f"{lab_name}_candidates.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            for rank, cand in enumerate(candidates, start=1):
                agreement = partition_agreement(cand.labels, truth)
                writer.writerow(
                    {
                        "rank": _fmt(rank),
                        "lab_name": lab_name,
                        "threshold": _fmt(cand.threshold),
                        "eigenvalue2": _fmt(cand.eigenvalue2),
                        "score": _fmt(cand.score),
                        "metastability": _fmt(cand.metastability),
                        "rm": _fmt(cand.rm),
                        "size0": _fmt(cand.size0),
                        "size1": _fmt(cand.size1),
                        "agreement_to_truth": _fmt(agreement),
                    }
                )

        best_agreement = 0.0
        if candidates:
            best_agreement = partition_agreement(candidates[0].labels, truth)

        summary[f"{lab_name}_best_agreement"] = float(best_agreement)
        summary[f"{lab_name}_n_candidates"] = int(len(candidates))
        summary[f"{lab_name}_top3_scores"] = [float(c.score) for c in candidates[:3]]

        print(f"{lab_name} best_agreement={_fmt(best_agreement)} n_candidates={len(candidates)}")

    with (out_dir / "summary.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, sort_keys=True, indent=2)
        fh.write("\n")


def _fmt(value: Any) -> str:
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
