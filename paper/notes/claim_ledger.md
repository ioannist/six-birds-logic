# Claim Ledger

All numeric claims in this ledger are frozen from `results/final_claims/` and/or its upstream frozen source files. No values here are to be recomputed in later writing tickets. Manuscript prose must cite these ledger anchors.

## Headline Claims

| Claim ID | Draftable claim | Exact value(s) | Source file | Exact key / row anchor | Intended manuscript section | Intended figure / table |
|---|---|---|---|---|---|---|
| CL1 | Parity beats the median random binary partition across the tested grid. | `parity_win_rate = 1.0` | `results/final_claims/claims.json` | `parity_win_rate` | `04_results_parity` | `figure_parity_robustness.csv` |
| CL2 | At `p_leak=0.05, tau=1`, parity RM is below the median random RM. | `parity_rep_p005_tau1_rm = 1.1102230246251565e-16`; `random_rep_p005_tau1_median_rm = 0.004999999999999977` | `results/final_claims/claims.json` | `parity_rep_p005_tau1_rm`; `random_rep_p005_tau1_median_rm` | `04_results_parity` | `figure_parity_robustness.csv` row `p_leak=0.05, tau=1` |
| CL3 | At `p_leak=0.20, tau=1`, parity RM remains below the median random RM. | `parity_rep_p020_tau1_rm = 0.0`; `random_rep_p020_tau1_median_rm = 0.020000000000000018` | `results/final_claims/claims.json` | `parity_rep_p020_tau1_rm`; `random_rep_p020_tau1_median_rm` | `04_results_parity` | `figure_parity_robustness.csv` row `p_leak=0.20, tau=1` |
| CL4 | The gate phase bundle contains a complete 108-row grid with no monotonicity anomalies. | `phase_n_rows = 108`; `phase_anomaly_count = 0` | `results/final_claims/claims.json` | `phase_n_rows`; `phase_anomaly_count` | `05_results_phase` | `figure_gate_phase.csv` |
| CL5 | NOT truth error increases from low to high output-noise settings. | `phase_not_err_truth_p0 = 0.02364324663553814`; `phase_not_err_truth_p01 = 0.11891459730843051` | `results/final_claims/claims.json` | `phase_not_err_truth_p0`; `phase_not_err_truth_p01` | `05_results_phase` | `figure_gate_phase.csv` |
| CL6 | CNOT truth error increases from low to high output-noise settings. | `phase_cnot_err_truth_p0 = 0.3333333333333333`; `phase_cnot_err_truth_p01 = 0.36666666666666675` | `results/final_claims/claims.json` | `phase_cnot_err_truth_p0`; `phase_cnot_err_truth_p01` | `05_results_phase` | `figure_gate_phase.csv` |
| CL7 | AND truth error increases from low to high output-noise settings. | `phase_and_err_truth_p0 = 0.022459509677580657`; `phase_and_err_truth_p01 = 0.11796760774206454` | `results/final_claims/claims.json` | `phase_and_err_truth_p0`; `phase_and_err_truth_p01` | `05_results_phase` | `figure_gate_phase.csv` |
| CL8 | Erased XOR loses much more input information than retained CNOT macro view. | `rev_vs_erased_ratio_unretained_input_info = 8.070108626770319` | `results/final_claims/claims.json` | `rev_vs_erased_ratio_unretained_input_info` | `06_results_reversible` | `table_reversible_vs_erased.csv` |
| CL9 | Erased XOR shows a much larger closure defect than retained CNOT macro view. | `rev_vs_erased_closure_defect_delta = 0.96` | `results/final_claims/claims.json` | `rev_vs_erased_closure_defect_delta` | `06_results_reversible` | `table_reversible_vs_erased.csv` |
| CL10 | Erased XOR shows a much larger RM than retained CNOT macro view. | `rev_vs_erased_rm_delta = 0.9599999999999999` | `results/final_claims/claims.json` | `rev_vs_erased_rm_delta` | `06_results_reversible` | `table_reversible_vs_erased.csv` |
| CL11 | Unsupervised partition discovery recovers the NOT-lab packaged bit exactly. | `discovery_not_best_agreement = 1.0` | `results/final_claims/claims.json` | `discovery_not_best_agreement` | `07_results_discovery` | `table_partition_discovery.csv` |
| CL12 | Unsupervised partition discovery recovers the parity packaged bit exactly. | `discovery_parity_best_agreement = 1.0` | `results/final_claims/claims.json` | `discovery_parity_best_agreement` | `07_results_discovery` | `table_partition_discovery.csv` |
| CL13 | The discovered NOT gate has truth table `[0, 1, 1, 0]` with low error at moderate noise. | `not_gate_truth_table_matrix_flat = [0, 1, 1, 0]`; `not_gate_error = 0.04845`; `not_gate_entropy = 0.2797739049244776`; `not_gate_sample_size = 20000` | `results/final_claims/claims.json` | `not_gate_truth_table_matrix_flat`; `not_gate_error`; `not_gate_entropy`; `not_gate_sample_size` | `07_results_discovery` | `table_gate_discovery.csv` |

## Result-Family Anchors

### Parity robustness
- Frozen files: `results/final_claims/claims.json`, `results/final_claims/figure_parity_robustness.csv`.
- Frozen table size: `figure_parity_robustness.csv` has 12 data rows.
- Drafting note: parity claims should cite both headline scalar anchors (CL1-CL3) and corresponding row-level values in `figure_parity_robustness.csv`.

### Gate phase diagram
- Frozen files: `results/final_claims/claims.json`, `results/final_claims/figure_gate_phase.csv`.
- Frozen table size: `figure_gate_phase.csv` has 108 data rows.
- Caution: multi-step CNOT (`tau > 1`) must be framed as a closure stress test, not as repeated application of a static truth-table object.

### Reversible vs erased
- Frozen files: `results/final_claims/claims.json`, `results/final_claims/table_reversible_vs_erased.csv`.
- Frozen table size: `table_reversible_vs_erased.csv` has 3 data rows.
- Caution: main accounting comparison should be phrased in terms of unretained input information, not only entropy drop.

### Partition discovery
- Frozen files: `results/final_claims/claims.json`, `results/final_claims/table_partition_discovery.csv`.
- Frozen table size: `table_partition_discovery.csv` has 2 data rows.
- Caution: discovery results should be presented as recovery from transition structure, not as externally declared symbolic labels.

### Gate discovery
- Frozen files: `results/final_claims/claims.json`, `results/final_claims/table_gate_discovery.csv`.
- Frozen table size: `table_gate_discovery.csv` has 1 data row.
- Drafting note: CL13 should be reported as the frozen moderate-noise NOT demonstration, with table values quoted exactly.

## Section Mapping

| Manuscript section file | Claim IDs |
|---|---|
| `04_results_parity` | `CL1-CL3` |
| `05_results_phase` | `CL4-CL7` |
| `06_results_reversible` | `CL8-CL10` |
| `07_results_discovery` | `CL11-CL13` |
