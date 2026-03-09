"""Render manuscript figure/table assets from frozen final-claims data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_RELATIVE_INPUTS = [
    "final_claims/figure_parity_robustness.csv",
    "final_claims/figure_gate_phase.csv",
    "final_claims/table_reversible_vs_erased.csv",
    "final_claims/table_partition_discovery.csv",
    "final_claims/table_gate_discovery.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render paper assets from frozen claims.")
    parser.add_argument("--results-dir", default="results", help="Results directory.")
    parser.add_argument("--paper-dir", default="paper", help="Paper directory.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = resolve_path(args.results_dir, repo_root)
    paper_dir = resolve_path(args.paper_dir, repo_root)

    ensure_required_inputs(results_dir)

    figures_dir = paper_dir / "figures"
    tables_dir = paper_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    parity_rows = read_csv_rows(results_dir / "final_claims" / "figure_parity_robustness.csv")
    phase_rows = read_csv_rows(results_dir / "final_claims" / "figure_gate_phase.csv")
    reversible_rows = read_csv_rows(results_dir / "final_claims" / "table_reversible_vs_erased.csv")
    partition_rows = read_csv_rows(results_dir / "final_claims" / "table_partition_discovery.csv")
    gate_rows = read_csv_rows(results_dir / "final_claims" / "table_gate_discovery.csv")

    render_parity_robustness(parity_rows, figures_dir / "parity_robustness.pdf")
    render_gate_phase(phase_rows, figures_dir / "gate_phase.pdf")

    write_reversible_table(reversible_rows, tables_dir / "reversible_vs_erased.tex")
    write_partition_table(partition_rows, tables_dir / "partition_discovery.tex")
    write_gate_table(gate_rows, tables_dir / "gate_discovery.tex")


def resolve_path(path_arg: str, repo_root: Path) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def ensure_required_inputs(results_dir: Path) -> None:
    missing = [rel for rel in REQUIRED_RELATIVE_INPUTS if not (results_dir / rel).is_file()]
    if missing:
        lines = "\n".join(f"  - {results_dir / rel}" for rel in missing)
        raise FileNotFoundError(f"Missing required frozen source files:\n{lines}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def render_parity_robustness(rows: list[dict[str, str]], out_path: Path) -> None:
    tau_values = [1, 2, 4]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True, constrained_layout=True)

    for idx, tau in enumerate(tau_values):
        ax = axes[idx]
        subset = [row for row in rows if int(float(row["tau"])) == tau]
        subset.sort(key=lambda row: float(row["p_leak"]))
        x = np.array([float(row["p_leak"]) for row in subset], dtype=float)
        y_parity = np.array([float(row["parity_rm"]) for row in subset], dtype=float)
        y_random = np.array([float(row["median_random_rm"]) for row in subset], dtype=float)
        y_and = np.array([float(row["and_rm"]) for row in subset], dtype=float)

        ax.plot(x, y_parity, marker="o", linewidth=1.5, label="parity")
        ax.plot(x, y_random, marker="s", linewidth=1.5, label="median random")
        ax.plot(x, y_and, marker="^", linewidth=1.5, label="AND")
        ax.set_title(rf"$\tau={tau}$")
        ax.set_xlabel(r"$p_{\mathrm{leak}}$")
        ax.grid(alpha=0.25, linewidth=0.4)
        if idx == 0:
            ax.set_ylabel("RM")
        ax.legend(loc="best", fontsize=8, frameon=False)

    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def render_gate_phase(rows: list[dict[str, str]], out_path: Path) -> None:
    gate_order = ["not", "cnot", "and"]
    barrier_order = [0.0, 2.0, 6.0]
    tau_order = [1, 2, 4]
    p_gate_order = [0.0, 0.02, 0.05, 0.1]

    row_order: list[tuple[str, float, int]] = [
        (gate, barrier, tau)
        for gate in gate_order
        for barrier in barrier_order
        for tau in tau_order
    ]

    row_labels: list[str] = []
    for gate, barrier, tau in row_order:
        gate_label = gate.upper() if gate != "and" else "AND"
        row_labels.append(f"{gate_label}, b={format_num(barrier)}, τ={tau}")

    lookup: dict[tuple[str, float, int, float], dict[str, str]] = {}
    for row in rows:
        key = (
            row["gate_name"].strip().lower(),
            float(row["barrier"]),
            int(float(row["tau"])),
            float(row["p_gate"]),
        )
        lookup[key] = row

    metrics = [
        ("err_truth", "truth error"),
        ("H_out_given_in", r"$H(\mathrm{out}\mid\mathrm{in})$"),
        ("rm_output", "RM"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 9.0), constrained_layout=True)
    x_tick_labels = [format_num(val) for val in p_gate_order]

    for ax, (metric_key, title) in zip(axes, metrics):
        matrix = np.zeros((len(row_order), len(p_gate_order)), dtype=float)
        for i, (gate, barrier, tau) in enumerate(row_order):
            for j, p_gate in enumerate(p_gate_order):
                key = (gate, barrier, tau, p_gate)
                if key not in lookup:
                    raise ValueError(f"Missing gate-phase row for {key}.")
                matrix[i, j] = float(lookup[key][metric_key])

        im = ax.imshow(matrix, aspect="auto", interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(p_gate_order)))
        ax.set_xticklabels(x_tick_labels)
        ax.set_xlabel(r"$p_{\mathrm{gate}}$")
        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels(row_labels, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def write_reversible_table(rows: list[dict[str, str]], out_path: Path) -> None:
    closure_rows: list[str] = []
    info_rows: list[str] = []
    for row in rows:
        label = format_view_label(row["view"])
        closure_rows.append(
            " & ".join(
                [
                    label,
                    row["n_states_view"],
                    latex_number(row["closure_defect"], places=2, sci=True),
                    latex_number(row["rm_view"], places=2, sci=True),
                    latex_number(row["epr_view"], places=2, sci=True),
                ]
            )
            + r" \\"
        )
        info_rows.append(
            " & ".join(
                [
                    label,
                    latex_number(row["H_in"], places=0),
                    latex_number(row["H_out"], places=0),
                    latex_number(row["H_out_given_in"], places=3),
                    latex_number(row["I_in_out"], places=3),
                    latex_number(row["entropy_drop"], places=0),
                    latex_number(row["unretained_input_info"], places=3),
                ]
            )
            + r" \\"
        )

    lines = [
        r"{\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}X r r r r@{}}",
        r"\toprule",
        r"\multicolumn{5}{@{}l}{\textit{Closure and audit metrics}} \\",
        r"\midrule",
        r"{View} & {States} & {Defect} & {RM} & {EPR} \\",
        r"\midrule",
        *closure_rows,
        r"\bottomrule",
        r"\end{tabularx}",
        "",
        r"\medskip",
        "",
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}X r r r r r r@{}}",
        r"\toprule",
        r"\multicolumn{7}{@{}l}{\textit{Channel-information metrics}} \\",
        r"\midrule",
        r"{View} & {$H_{\mathrm{in}}$} & {$H_{\mathrm{out}}$} & {\makecell{$H(\mathrm{out}\mid$\\$\mathrm{in})$}} & {\makecell{$I(\mathrm{in};$\\$\mathrm{out})$}} & {$\Delta H$} & {$U_{\text{loss}}$} \\",
        r"\midrule",
        *info_rows,
        r"\bottomrule",
        r"\end{tabularx}",
        r"}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_partition_table(rows: list[dict[str, str]], out_path: Path) -> None:
    table_rows: list[str] = []
    for row in rows:
        table_rows.append(
            " & ".join(
                [
                    format_lab_label(row["lab_name"]),
                    latex_number(row["best_agreement"], places=0),
                    latex_number(row["n_candidates"], places=0),
                    latex_number(row["best_score"], places=3),
                    latex_number(row["best_metastability"], places=3),
                    latex_number(row["best_rm"], places=2, sci=True),
                    latex_number(row["top2_score"], places=3),
                    latex_number(row["top3_score"], places=3),
                ]
            )
            + r" \\"
        )

    lines = [
        r"{\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l r r r r r r r}",
        r"\toprule",
        r"{Lab} & {Agree.} & {Cand.} & {Score} & {Meta.} & {RM} & {2nd} & {3rd} \\",
        r"\midrule",
        *table_rows,
        r"\bottomrule",
        r"\end{tabular*}",
        r"}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_gate_table(rows: list[dict[str, str]], out_path: Path) -> None:
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one gate-discovery row, found {len(rows)}.")
    row = rows[0]
    lines = [
        r"{\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabularx}{\linewidth}{@{}l X l X@{}}",
        r"\toprule",
        r"Metric & Value & Metric & Value \\",
        r"\midrule",
        rf"Gate & {format_gate_label(row['gate_name'])} & Truth bits & {texttt(row['truth_table_bits'])} \\",
        rf"Truth table & {texttt(row['truth_table_matrix_flat'])} & Error & {latex_number(row['error'], places=4)} \\",
        rf"Entropy & {latex_number(row['entropy'], places=3)} & Sample size & {latex_number(row['sample_size'], places=0)} \\",
        rf"Input score & {latex_number(row['input_partition_score'], places=3)} & $\Delta I_{{\mathrm{{out}}}}$ & {latex_number(row['output_delta_I'], places=3)} \\",
        rf"$I_{{\mathrm{{future}}}}$ & {latex_number(row['output_future_I'], places=3)} & $I_{{\mathrm{{current}}}}$ & {latex_number(row['output_current_I'], places=0)} \\",
        r"\bottomrule",
        r"\end{tabularx}",
        r"}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_latex_tabular(path: Path, headers: list[str], rows: Iterable[list[str]]) -> None:
    rows = list(rows)
    alignment = "l" * len(headers)
    lines = [
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(headers) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def texttt(value: str) -> str:
    escaped = (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("$", r"\$")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )
    return rf"\texttt{{{escaped}}}"


def latex_number(value: str, *, places: int, sci: bool = False) -> str:
    number = float(value)
    if sci:
        if abs(number) < 1e-14:
            return "0" if number == 0 else format_scientific(number, places)
        if abs(number) >= 1e-3:
            return format_decimal(number, places)
        return format_scientific(number, places)
    return format_decimal(number, places)


def format_decimal(number: float, places: int) -> str:
    if places == 0:
        return str(int(round(number)))
    return f"{number:.{places}f}"


def format_scientific(number: float, places: int) -> str:
    mantissa, exponent = f"{number:.{places}e}".split("e")
    exp = int(exponent)
    return rf"\({mantissa}\times 10^{{{exp}}}\)"


def format_view_label(value: str) -> str:
    labels = {
        "cnot_micro": "CNOT micro",
        "cnot_macro": "CNOT macro",
        "xor_erased_macro": "XOR erased macro",
    }
    return labels.get(value, value.replace("_", " "))


def format_lab_label(value: str) -> str:
    labels = {
        "not": "NOT",
        "parity_sector": "Parity sector",
    }
    return labels.get(value, value.replace("_", " "))


def format_gate_label(value: str) -> str:
    return "AND" if value.strip().lower() == "and" else value.strip().upper()


def format_num(x: float) -> str:
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return format(x, ".2g") if x < 0.1 else format(x, ".12g")


if __name__ == "__main__":
    main()
