"""Render manuscript figure/table assets from frozen final-claims data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


REQUIRED_RELATIVE_INPUTS = [
    "final_claims/figure_parity_robustness.csv",
    "final_claims/figure_gate_phase.csv",
    "final_claims/table_reversible_vs_erased.csv",
    "final_claims/table_partition_discovery.csv",
    "final_claims/table_gate_discovery.csv",
    "exp_gate_discovery/gates.json",
]

# Values below this magnitude are floating-point residue of quantities that are
# exactly zero in rational arithmetic (results/math_review/audit.json).
ROUNDOFF = 1e-12

# Validated categorical slots (blue, orange, aqua) and a single-hue blue ramp.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
SEQ_BLUE = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render paper assets from frozen claims.")
    parser.add_argument("--results-dir", default="results", help="Results directory.")
    parser.add_argument("--paper-dir", default="paper", help="Paper directory.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = resolve_path(args.results_dir, repo_root)
    paper_dir = resolve_path(args.paper_dir, repo_root)

    ensure_required_inputs(results_dir)
    configure_matplotlib()

    figures_dir = paper_dir / "figures"
    tables_dir = paper_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    parity_rows = read_csv_rows(results_dir / "final_claims" / "figure_parity_robustness.csv")
    phase_rows = read_csv_rows(results_dir / "final_claims" / "figure_gate_phase.csv")
    reversible_rows = read_csv_rows(results_dir / "final_claims" / "table_reversible_vs_erased.csv")
    partition_rows = read_csv_rows(results_dir / "final_claims" / "table_partition_discovery.csv")
    gate_rows = read_csv_rows(results_dir / "final_claims" / "table_gate_discovery.csv")
    with (results_dir / "exp_gate_discovery" / "gates.json").open("r", encoding="utf-8") as fh:
        gate_source = json.load(fh)

    render_parity_robustness(parity_rows, figures_dir / "parity_robustness.pdf")
    render_gate_phase(phase_rows, figures_dir / "gate_phase.pdf")
    render_information_budget(reversible_rows, figures_dir / "information_budget.pdf")

    write_reversible_table(reversible_rows, tables_dir / "reversible_vs_erased.tex")
    write_partition_table(partition_rows, tables_dir / "partition_discovery.tex")
    write_gate_table(gate_rows, gate_source, tables_dir / "gate_discovery.tex")


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


def configure_matplotlib() -> None:
    """Match the manuscript typeface when Latin Modern is installed."""
    family = "serif"
    for name in ("lmroman10-regular.otf", "lmroman10-bold.otf"):
        for path in font_manager.findSystemFonts() + _texlive_font_paths(name):
            if Path(path).name == name:
                font_manager.fontManager.addfont(path)
                family = "Latin Modern Roman"
                break
    plt.rcParams.update(
        {
            "font.family": family,
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 3,
        }
    )


def _texlive_font_paths(name: str) -> list[str]:
    import subprocess

    try:
        out = subprocess.run(["kpsewhich", name], capture_output=True, text=True, check=False)
    except OSError:
        return []
    path = out.stdout.strip()
    return [path] if path else []


def _style_axes(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.5)
    ax.set_axisbelow(True)


def render_parity_robustness(rows: list[dict[str, str]], out_path: Path) -> None:
    """Frozen RM values as markers; exact closed forms as curves (Section 4)."""
    tau_values = [1, 2, 4]
    p_grid = np.linspace(0.01, 0.22, 200)
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.55), sharey=True, constrained_layout=True)

    def exact_and(p: np.ndarray, tau: int) -> np.ndarray:
        return (2.0 / 3.0) * np.abs(0.5 - 1.2 * p) * np.abs(1.0 - 2.0 * p) ** (tau - 1)

    def exact_a(p: np.ndarray, tau: int) -> np.ndarray:
        return (p / 2.0) * np.abs(1.0 - 2.0 * p) ** (tau - 1)

    def exact_b(p: np.ndarray, tau: int) -> np.ndarray:
        return (p / 10.0) * np.abs(1.0 - 2.0 * p) ** (tau - 1)

    for idx, tau in enumerate(tau_values):
        ax = axes[idx]
        subset = [row for row in rows if int(float(row["tau"])) == tau]
        subset.sort(key=lambda row: float(row["p_leak"]))
        x = np.array([float(row["p_leak"]) for row in subset], dtype=float)
        y_random = np.array([float(row["median_random_rm"]) for row in subset], dtype=float)
        y_and = np.array([float(row["and_rm"]) for row in subset], dtype=float)
        y_parity = np.array([float(row["parity_rm"]) for row in subset], dtype=float)
        if np.any(np.abs(y_parity) > ROUNDOFF):
            raise ValueError("Parity RM exceeds the roundoff threshold; the figure annotation would be false.")

        ax.plot(p_grid, exact_and(p_grid, tau), color=AQUA, linewidth=1.6, label=r"AND lens $a\wedge b$")
        ax.plot(p_grid, exact_a(p_grid, tau), color=ORANGE, linewidth=1.6, label=r"bit lens $a$")
        ax.plot(p_grid, exact_b(p_grid, tau), color=BLUE, linewidth=1.6, label=r"bit lens $b$")
        ax.plot(x, y_and, linestyle="none", marker="o", markersize=4.2, color=AQUA,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.plot(x, y_random, linestyle="none", marker="s", markersize=4.2, color=BLUE,
                markeredgecolor="white", markeredgewidth=0.8, label="sampled median")

        ax.set_yscale("log")
        ax.set_ylim(8e-4, 1.0)
        ax.set_xlim(0.0, 0.225)
        ax.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20])
        ax.set_xticklabels(["0", "0.05", "0.10", "0.15", "0.20"])
        ax.set_title(rf"$\tau={tau}$", color=INK)
        ax.set_xlabel(r"leakage $p_{\mathrm{leak}}$")
        ax.text(0.97, 0.04, r"parity lens: $\mathrm{RM}=0$ exactly", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color=INK_2)
        _style_axes(ax)
        if idx == 0:
            ax.set_ylabel("route mismatch RM (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)

    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def render_gate_phase(rows: list[dict[str, str]], out_path: Path) -> None:
    gate_order = ["not", "and", "cnot"]
    barrier_order = [0.0, 2.0, 6.0]
    tau_order = [1, 2, 4]
    p_gate_order = [0.0, 0.02, 0.05, 0.1]

    lookup: dict[tuple[str, float, int, float], dict[str, str]] = {}
    for row in rows:
        key = (
            row["gate_name"].strip().lower(),
            float(row["barrier"]),
            int(float(row["tau"])),
            float(row["p_gate"]),
        )
        lookup[key] = row
    if len(lookup) != len(gate_order) * len(barrier_order) * len(tau_order) * len(p_gate_order):
        raise ValueError("Gate-phase grid is incomplete or duplicated.")

    metrics = [
        ("err_truth", "truth error", 0.5),
        ("H_out_given_in", r"$H(\mathrm{out}\mid\mathrm{in})$ [bits]", 1.0),
        ("rm_output", "output-lens RM", 1.0),
    ]
    row_labels = [f"barrier {format_num(barrier)}, " + rf"$\tau={tau}$"
                  for barrier in barrier_order for tau in tau_order]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

    fig, axes = plt.subplots(3, 3, figsize=(6.5, 7.4), constrained_layout=True)
    for i, gate in enumerate(gate_order):
        for j, (metric_key, title, vmax) in enumerate(metrics):
            ax = axes[i, j]
            matrix = np.zeros((len(row_labels), len(p_gate_order)), dtype=float)
            for r, (barrier, tau) in enumerate((b, t) for b in barrier_order for t in tau_order):
                for c, p_gate in enumerate(p_gate_order):
                    value = float(lookup[(gate, barrier, tau, p_gate)][metric_key])
                    matrix[r, c] = 0.0 if abs(value) < ROUNDOFF else value
            im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=vmax)
            for r in range(matrix.shape[0]):
                for c in range(matrix.shape[1]):
                    value = matrix[r, c]
                    text = "0" if value == 0.0 else ("<.01" if value < 0.005 else f"{value:.2f}".lstrip("0"))
                    ax.text(c, r, text, ha="center", va="center", fontsize=6.3,
                            color="white" if value > 0.55 * vmax else INK)
            for boundary in (2.5, 5.5):
                ax.axhline(boundary, color="white", linewidth=1.6)
            ax.set_xticks(np.arange(len(p_gate_order)))
            ax.set_xticklabels([format_num(v) for v in p_gate_order] if i == len(gate_order) - 1 else [])
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_yticklabels(row_labels if j == 0 else [], fontsize=7)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(title, color=INK)
            if i == len(gate_order) - 1:
                ax.set_xlabel(r"output noise $p_{\mathrm{gate}}$")
            if j == 0:
                ax.set_ylabel(format_gate_label(gate), fontsize=10, fontweight="bold", color=INK)
            if i == len(gate_order) - 1:
                cb = fig.colorbar(im, ax=axes[:, j], location="bottom", shrink=0.8, aspect=30, pad=0.01)
                cb.outline.set_visible(False)
                cb.ax.tick_params(labelsize=7, length=2)

    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def render_information_budget(rows: list[dict[str, str]], out_path: Path) -> None:
    """Split H_in into I and U_loss, and H_out into I and H(out|in), per view."""
    by_view = {row["view"]: row for row in rows}
    order = ["cnot_micro", "cnot_macro", "xor_erased_macro"]
    labels = ["micro readout\n(12 states)", "retained register\n(4 states)", "erased output\n(1 bit)"]
    mutual = np.array([float(by_view[v]["I_in_out"]) for v in order])
    loss = np.array([float(by_view[v]["unretained_input_info"]) for v in order])
    h_cond = np.array([float(by_view[v]["H_out_given_in"]) for v in order])
    h_in = np.array([float(by_view[v]["H_in"]) for v in order])
    h_out = np.array([float(by_view[v]["H_out"]) for v in order])
    if not (np.allclose(mutual + loss, h_in) and np.allclose(mutual + h_cond, h_out)):
        raise ValueError("Information identities fail for the reversible-versus-erased rows.")

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.35), sharey=True, constrained_layout=True)
    y = np.arange(len(order))[::-1]
    height = 0.56
    panels = [
        (axes[0], loss, ORANGE, r"unretained $U_{\mathrm{loss}}$", "input entropy $H_{\\mathrm{in}}$ [bits]"),
        (axes[1], h_cond, AQUA, r"output noise $H(\mathrm{out}\mid\mathrm{in})$", "output entropy $H_{\\mathrm{out}}$ [bits]"),
    ]
    for ax, second, color, second_label, xlabel in panels:
        ax.barh(y, mutual, height=height, color=BLUE, edgecolor="white", linewidth=1.0,
                label=r"shared $I(\mathrm{in};\mathrm{out})$")
        ax.barh(y, second, left=mutual, height=height, color=color, edgecolor="white", linewidth=1.0,
                label=second_label)
        for yy, m, s in zip(y, mutual, second):
            ax.text(m / 2.0, yy, f"{m:.3f}", ha="center", va="center", fontsize=7.5, color="white")
            if s > 0.35:
                ax.text(m + s / 2.0, yy, f"{s:.3f}", ha="center", va="center", fontsize=7.5, color=INK)
            else:
                ax.text(m + s + 0.05, yy, f"{s:.3f}", ha="left", va="center", fontsize=7.5, color=INK_2)
        ax.set_xlim(0.0, 3.9)
        ax.set_xlabel(xlabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color="#e4e3df", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_title("where the two input bits go", color=INK)
    axes[1].set_title("what the reported output contains", color=INK)

    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def table_number(value: str, *, places: int) -> str:
    """Decimal entry with exact zeros restored below the roundoff threshold."""
    number = float(value)
    if abs(number) < ROUNDOFF:
        return "0"
    if places == 0 or abs(number - round(number)) < 1e-12:
        return str(int(round(number))) if abs(number - round(number)) < 1e-12 else format_decimal(number, places)
    text = format_decimal(number, places)
    return text.replace("-", "$-$") if text.startswith("-") else text


def write_reversible_table(rows: list[dict[str, str]], out_path: Path) -> None:
    table_rows: list[str] = []
    for row in rows:
        table_rows.append(
            " & ".join(
                [
                    format_view_label(row["view"]),
                    row["n_states_view"],
                    table_number(row["closure_defect"], places=2),
                    table_number(row["rm_view"], places=2),
                    table_number(row["epr_view"], places=2),
                    table_number(row["H_out"], places=3),
                    table_number(row["H_out_given_in"], places=3),
                    table_number(row["I_in_out"], places=3),
                    table_number(row["entropy_drop"], places=3),
                    table_number(row["unretained_input_info"], places=3),
                ]
            )
            + r" \\"
        )

    lines = [
        r"{\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l r r r r r r r r r@{}}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Closure and audit} & \multicolumn{5}{c}{Channel information [bits]} \\",
        r"\cmidrule(lr){3-5}\cmidrule(l){6-10}",
        r"View & States & $\Delta$ & RM & EPR & $H_{\mathrm{out}}$ & $H(\mathrm{out}\mid\mathrm{in})$ & $I$ & $\Delta H$ & $U_{\mathrm{loss}}$ \\",
        r"\midrule",
        *table_rows,
        r"\bottomrule",
        r"\end{tabular*}",
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
                    latex_number(row["best_metastability"], places=4),
                    table_number(row["best_rm"], places=2),
                    latex_number(row["best_score"], places=4),
                    latex_number(row["top2_score"], places=3),
                    latex_number(row["top3_score"], places=3),
                ]
            )
            + r" \\"
        )

    lines = [
        r"{\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l r r r r r r r@{}}",
        r"\toprule",
        r" & & & \multicolumn{3}{c}{Top-ranked cut} & \multicolumn{2}{c}{Runner-up scores} \\",
        r"\cmidrule(lr){4-6}\cmidrule(l){7-8}",
        r"Laboratory & Agreement & Candidates & $S_\tau$ & RM & Score & 2nd & 3rd \\",
        r"\midrule",
        *table_rows,
        r"\bottomrule",
        r"\end{tabular*}",
        r"}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_gate_table(rows: list[dict[str, str]], source: dict, out_path: Path) -> None:
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one gate-discovery row, found {len(rows)}.")
    row = rows[0]
    exact = source[row["gate_name"].strip().lower()]
    if abs(float(exact["error"]) - float(row["error"])) > 1e-12:
        raise ValueError("Gate-discovery source and frozen claims disagree on the sampled error.")
    confusion = exact["confusion"]
    lines = [
        r"{\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}} l c c@{}}",
        r"\toprule",
        r"Quantity & Exact induced channel & Balanced sample \\",
        r"\midrule",
        rf"Truth bits (declared orientation) & {texttt(json.dumps(exact['exact_truth_table_bits']).replace(' ', ''))} & {texttt(row['truth_table_bits'])} \\",
        rf"Gate error & {exact['exact_error']:.3f} & {float(row['error']):.5f} \\",
        rf"$H(\mathrm{{out}}\mid\mathrm{{in}})$ [bits] & {exact['exact_entropy']:.4f} & {float(row['entropy']):.4f} \\",
        rf"Transitions (input 0 $\to$ 0/1; input 1 $\to$ 0/1) & -- & {confusion[0][0]}/{confusion[0][1]}; {confusion[1][0]}/{confusion[1][1]} \\",
        r"\midrule",
        rf"Input partition score $S_\tau-\mathrm{{RM}}$ & \multicolumn{{2}}{{c}}{{{float(row['input_partition_score']):.5f}}} \\",
        rf"Output partition $I_{{\mathrm{{future}}}}$ / $I_{{\mathrm{{current}}}}$ [bits] & \multicolumn{{2}}{{c}}{{{float(row['output_future_I']):.4f} / {table_number(row['output_current_I'], places=0)}}} \\",
        r"\bottomrule",
        r"\end{tabular*}",
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
        "cnot_micro": "Micro readout",
        "cnot_macro": "Retained register",
        "xor_erased_macro": "Erased output",
    }
    return labels.get(value, value.replace("_", " "))


def format_lab_label(value: str) -> str:
    labels = {
        "not": "NOT",
        "parity_sector": "Parity-sector",
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
