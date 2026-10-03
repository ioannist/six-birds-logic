import LogicClosure.QuotientDynamics

namespace LogicClosure

abbrev Z4 := Fin 4
abbrev X2 := Fin 2

def parityLens (i : Z4) : X2 :=
  ⟨i.1 % 2, Nat.mod_lt _ (by decide)⟩

def parityClass : Z4 → Nat := fun i => i.1 % 2

def paritySetoid : Setoid Z4 := kernelSetoid parityClass

-- Connect the setoid used below to the actual Fin 2-valued lens, rather
-- than leaving the example as a quotient of an unrelated Nat-valued map.
theorem paritySetoid_iff_lens_eq (a b : Z4) :
    paritySetoid.r a b ↔ parityLens a = parityLens b := by
  constructor
  · intro h
    apply Fin.ext
    exact h
  · intro h
    exact congrArg Fin.val h

def collapseToParityRep (i : Z4) : Z4 :=
  ⟨i.1 % 2, by
    have hmod : i.1 % 2 < 2 := Nat.mod_lt _ (by decide)
    exact Nat.lt_trans hmod (by decide)⟩

theorem collapseToParityRep_respects :
    Respects paritySetoid collapseToParityRep := by
  intro a b hab
  dsimp [paritySetoid, kernelSetoid, parityClass, collapseToParityRep] at hab ⊢
  simpa [Nat.mod_mod] using hab

theorem collapseToParityRep_preserves_lens (i : Z4) :
    parityLens (collapseToParityRep i) = parityLens i := by
  apply Fin.ext
  simp [parityLens, collapseToParityRep]

theorem parityLens_surjective (x : X2) : ∃ i : Z4, parityLens i = x := by
  refine ⟨⟨x.val, Nat.lt_trans x.isLt (by decide)⟩, ?_⟩
  apply Fin.ext
  exact Nat.mod_eq_of_lt x.isLt

def parityInduced : Quotient paritySetoid → Quotient paritySetoid :=
  inducedMap paritySetoid collapseToParityRep collapseToParityRep_respects

theorem parityInduced_eq_id : parityInduced = id := by
  funext q
  induction q using Quotient.ind with
  | _ a =>
    apply Quotient.sound
    exact (paritySetoid_iff_lens_eq _ _).mpr (collapseToParityRep_preserves_lens a)

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
