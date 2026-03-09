# Figure and Table Inventory

## Planned Main-Text Assets

| Asset ID | Planned title | Type | Source file | Rows / shape | Required columns | Intended manuscript section | Narrative role | Status |
|---|---|---|---|---|---|---|---|---|
| A1 | Logic as an audited closure of packaged predicates | Figure | conceptual / not data-derived | n/a | n/a | `02_framework` or `03_diagnostics_methods` | One-glance conceptual overview | planned |
| A2 | Parity robustness against random partitions | Figure | `results/final_claims/figure_parity_robustness.csv` | 12 data rows | `p_leak, tau, parity_rm, and_rm, median_random_rm, win_vs_random_median` | `04_results_parity` | Show parity advantage over baseline lenses across grid | frozen data ready |
| A3 | Gate closure phase diagram across NOT/CNOT/AND | Figure | `results/final_claims/figure_gate_phase.csv` | 108 data rows | `gate_name, barrier, p_gate, tau, err_truth, err_induced, H_out_given_in, rm_output` | `05_results_phase` | Show noise/barrier/tau trends in induced gate quality and closure | frozen data ready |
| A4 | Reversible embedding versus erased XOR audit table | Table | `results/final_claims/table_reversible_vs_erased.csv` | 3 data rows | `view, n_states_view, closure_defect, rm_view, epr_view, I_in_out, unretained_input_info` | `06_results_reversible` | Quantify retained-vs-erased gap in information and closure | frozen data ready |
| A5 | Partition discovery outcomes | Table | `results/final_claims/table_partition_discovery.csv` | 2 data rows | `lab_name, best_agreement, n_candidates, best_score, best_metastability, best_rm` | `07_results_discovery` | Summarize unsupervised recovery quality for NOT and parity labs | frozen data ready |
| A6 | NOT gate discovery summary | Table | `results/final_claims/table_gate_discovery.csv` | 1 data row | `gate_name, truth_table_matrix_flat, error, entropy, sample_size, output_delta_I` | `07_results_discovery` | Freeze discovered operator statistics for manuscript claim | frozen data ready |

## Frozen Data Assets

- `results/final_claims/figure_parity_robustness.csv` — 12 data rows — parity robustness figure data for Section `04_results_parity`.
- `results/final_claims/figure_gate_phase.csv` — 108 data rows — phase-diagram figure data for Section `05_results_phase`.
- `results/final_claims/table_reversible_vs_erased.csv` — 3 data rows — retained-vs-erased comparison table for Section `06_results_reversible`.
- `results/final_claims/table_partition_discovery.csv` — 2 data rows — partition discovery summary table for Section `07_results_discovery`.
- `results/final_claims/table_gate_discovery.csv` — 1 data row — NOT gate discovery summary table for Section `07_results_discovery`.

## Notes for Rendering Tickets

Later rendering tickets must not recompute values. All figures/tables should be pure exports from `results/final_claims/`. Captions should be written in manuscript drafting tickets, not in this inventory note.
