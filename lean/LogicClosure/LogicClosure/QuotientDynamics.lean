namespace LogicClosure

def Respects {α : Type _} (r : Setoid α) (F : α → α) : Prop :=
  ∀ ⦃a b : α⦄, r.r a b → r.r (F a) (F b)

def inducedMap {α : Type _} (r : Setoid α) (F : α → α) (hF : Respects r F) :
    Quotient r → Quotient r :=
  Quotient.lift
    (fun a => Quotient.mk r (F a))
    (by
      intro a b hab
      exact Quotient.sound (hF hab))

theorem inducedMap_sound
    {α : Type _} {r : Setoid α} {F : α → α} (hF : Respects r F)
    {a b : α} (hab : r.r a b) :
    inducedMap r F hF (Quotient.mk r a) = inducedMap r F hF (Quotient.mk r b) := by
  change Quotient.mk r (F a) = Quotient.mk r (F b)
  exact Quotient.sound (hF hab)

theorem inducedMap_mk
    {α : Type _} (r : Setoid α) (F : α → α) (hF : Respects r F) (a : α) :
    inducedMap r F hF (Quotient.mk r a) = Quotient.mk r (F a) := by
  rfl

-- Conversely, a commuting quotient update forces representative consistency.
theorem respects_iff_exists_quotientMap
    {α : Type _} (r : Setoid α) (F : α → α) :
    Respects r F ↔ ∃ g : Quotient r → Quotient r,
      ∀ a, g (Quotient.mk r a) = Quotient.mk r (F a) := by
  constructor
  · intro hF
    exact ⟨inducedMap r F hF, inducedMap_mk r F hF⟩
  · rintro ⟨g, hg⟩ a b hab
    apply Quotient.exact
    rw [← hg a, ← hg b]
    exact congrArg g (Quotient.sound hab)

theorem inducedMap_unique
    {α : Type _} {r : Setoid α} {F : α → α} (hF : Respects r F)
    {g : Quotient r → Quotient r}
    (hg : ∀ a, g (Quotient.mk r a) = Quotient.mk r (F a)) :
    g = inducedMap r F hF := by
  funext q
  induction q using Quotient.ind with
  | _ a => exact hg a

theorem respects_comp
    {α : Type _} {r : Setoid α} {F G : α → α}
    (hF : Respects r F) (hG : Respects r G) : Respects r (G ∘ F) := by
  intro a b hab
  exact hG (hF hab)

theorem inducedMap_comp
    {α : Type _} {r : Setoid α} {F G : α → α}
    (hF : Respects r F) (hG : Respects r G) :
    inducedMap r (G ∘ F) (respects_comp hF hG) =
      inducedMap r G hG ∘ inducedMap r F hF := by
  funext q
  induction q using Quotient.ind with
  | _ a => rfl

section Examples

def kernelSetoid {α β : Type _} (f : α → β) : Setoid α where
  r := fun a b => f a = f b
  iseqv := by
    refine ⟨?_, ?_, ?_⟩
    · intro a
      rfl
    · intro a b hab
      exact hab.symm
    · intro a b c hab hbc
      exact hab.trans hbc

def firstBitSetoid : Setoid (Bool × Bool) :=
  kernelSetoid (fun p => p.1)

def flipSecond (p : Bool × Bool) : Bool × Bool :=
  (p.1, !p.2)

theorem flipSecond_respects : Respects firstBitSetoid flipSecond := by
  intro a b hab
  simpa [firstBitSetoid, kernelSetoid, flipSecond] using hab

example : Quotient firstBitSetoid → Quotient firstBitSetoid :=
  inducedMap firstBitSetoid flipSecond flipSecond_respects

example :
    inducedMap firstBitSetoid flipSecond flipSecond_respects
        (Quotient.mk firstBitSetoid (true, false)) =
      Quotient.mk firstBitSetoid (true, true) := by
  simpa [flipSecond] using
    inducedMap_mk firstBitSetoid flipSecond flipSecond_respects (true, false)

end Examples

end LogicClosure
