# Six Birds: Logic Instantiation

This repository contains the **logic instantiation** for the paper:

> **To XOR a Stone with Six Birds: Closure Diagnostics for Emergent Bits, Gates, and Booleanity**
>
> Archived at: https://zenodo.org/records/18919370
>
> DOI: https://doi.org/10.5281/zenodo.18919370

This paper is the logic-focused instantiation of the emergence calculus introduced in *Six Birds: Foundations of Emergence Calculus*. It treats logic as something a substrate can earn at a layer rather than something given in advance, and develops auditable closure diagnostics for when packaged distinctions behave like bits, gates, and Boolean operators under coarse-grained dynamics.

## What this repository provides

The logic instantiation implements:

- **Finite Markov laboratories**: controlled micro-to-macro testbeds for packaged predicates, stable carriers, induced operators, and closure audits
- **Logic-layer diagnostics**: metastability, route mismatch, closure defect, gate error, conditional entropy, unretained input information, and entropy-production audits
- **Parity robustness experiments**: showing that parity/XOR-type variables are unusually strong coarse variables across the tested grid
- **Gate-phase experiments**: stress-testing NOT, AND, and reversible XOR embeddings under noise, staging barriers, and timescale variation
- **Reversible-versus-erased comparisons**: quantifying how coarse-graining a reversible embedding into an erased XOR output degrades closure and increases hidden information loss
- **Discovery pipeline**: recovering stable packaged bits and reconstructing a NOT gate directly from transition structure
- **Artifact contract**: manuscript figures, tables, and headline claims are generated from frozen CSV/JSON outputs under `results/final_claims/`, with paper assets rendered from that bundle

## Scope and limitations

The paper is explicit about what it does and does not establish:

- The experiments are controlled diagnostics, not claims that Boolean logic is fundamental or universally privileged
- Closure scores are operational stress tests for candidate logic layers, not metaphysical proofs that a substrate "really is" logical
- The reversible-versus-erased comparison diagnoses retained versus discarded information under coarse-graining; it does not claim a new physical law beyond the audited setting
- The discovery pipeline is intentionally minimal and falsifiable; it shows that bits and a NOT gate can be recovered from transition structure in the laboratory setting, not that arbitrary natural systems will yield the same result automatically

## Install

```bash
pip install -e .[dev]
cd lean/LogicClosure && lake build
```

## Test

```bash
make test
PYTHONPATH=src python scripts/validate_final_state.py
```

## Run experiments

```bash
python experiments/run_sweep.py --config configs/exp_parity_vs_and.json
python experiments/run_parity_robustness.py --config configs/exp_parity_robustness.json
python experiments/run_gate_phase_diagram.py --config configs/exp_gate_phase_diagram.json
python experiments/run_reversible_vs_erased.py --config configs/exp_reversible_vs_erased.json
python experiments/run_discovery_smoke.py
python experiments/run_gate_discovery.py --config configs/exp_gate_discovery.json
```

To reproduce the full frozen bundle in one pass:

```bash
make reproduce-all
```

## Build paper

```bash
make paper
```

## Seal final artifact bundle

```bash
make seal
```
