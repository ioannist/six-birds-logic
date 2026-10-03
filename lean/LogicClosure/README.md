# LogicClosure

This Lean 4.28.0 project proves the exact predicate and quotient core of the
logic laboratory. It does not formalize floating-point experiments or infer
metastability from definability.

- `Definable.lean`: factorization iff fiber constancy, including Boolean-valued
  predicates; closure under the Boolean connectives; pullback through an
  explicitly descended update. No surjectivity of the lens is assumed.
- `QuotientDynamics.lean`: quotient descent iff representative consistency,
  uniqueness, and compatibility with composition.
- `FiniteExample.lean`: the actual `Fin 4 → Fin 2` parity lens, its kernel
  setoid, and the induced identity update.
- `GateExamples.lean`: ideal register updates, reversible CNOT, failure of
  one-step target-only descent, and recovery of descent after two ticks.

From this directory, run `lake build`. From the repository root, run
`python3 scripts/check_lean_axioms.py` to build and inspect all named theorems'
transitive axioms. Only the standard Lean foundations `propext`, `Quot.sound`,
and `Classical.choice` are permitted. The mathematical review and numerical
coverage limits are documented in `docs/mathematical_review.txt`.
