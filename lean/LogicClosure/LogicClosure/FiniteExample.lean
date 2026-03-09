import LogicClosure.QuotientDynamics

namespace LogicClosure

abbrev Z4 := Fin 4
abbrev X2 := Fin 2

def parityLens (i : Z4) : X2 :=
  ⟨i.1 % 2, Nat.mod_lt _ (by decide)⟩

def parityClass : Z4 → Nat := fun i => i.1 % 2

def paritySetoid : Setoid Z4 := kernelSetoid parityClass

def collapseToParityRep (i : Z4) : Z4 :=
  ⟨i.1 % 2, by
    have hmod : i.1 % 2 < 2 := Nat.mod_lt _ (by decide)
    exact Nat.lt_trans hmod (by decide)⟩

theorem collapseToParityRep_respects :
    Respects paritySetoid collapseToParityRep := by
  intro a b hab
  dsimp [paritySetoid, kernelSetoid, parityClass, collapseToParityRep] at hab ⊢
  simpa [Nat.mod_mod] using hab

def parityInduced : Quotient paritySetoid → Quotient paritySetoid :=
  inducedMap paritySetoid collapseToParityRep collapseToParityRep_respects

section Examples

example : Quotient paritySetoid → Quotient paritySetoid := parityInduced

example :
    parityInduced (Quotient.mk paritySetoid ⟨3, by decide⟩) =
      Quotient.mk paritySetoid ⟨1, by decide⟩ := by
  simpa [parityInduced, collapseToParityRep] using
    inducedMap_mk paritySetoid collapseToParityRep collapseToParityRep_respects
      ⟨3, by decide⟩

example :
    parityInduced (Quotient.mk paritySetoid ⟨0, by decide⟩) =
      parityInduced (Quotient.mk paritySetoid ⟨2, by decide⟩) := by
  apply inducedMap_sound collapseToParityRep_respects
  show parityClass ⟨0, by decide⟩ = parityClass ⟨2, by decide⟩
  simp [parityClass]

end Examples

end LogicClosure
