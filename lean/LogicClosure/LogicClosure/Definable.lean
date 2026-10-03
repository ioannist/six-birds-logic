namespace LogicClosure

def fiberEq {Z X : Type _} (f : Z → X) (z z' : Z) : Prop := f z = f z'

def Definable {Z X : Type _} (f : Z → X) (p : Z → Prop) : Prop :=
  ∃ q : X → Prop, p = q ∘ f

def FiberConstant {Z X : Type _} (f : Z → X) (p : Z → Prop) : Prop :=
  ∀ ⦃z z' : Z⦄, fiberEq f z z' → (p z ↔ p z')

-- No surjectivity or inhabitance assumptions are needed: the extension is
-- false on labels outside the image of the lens.
theorem definable_iff_fiberConstant
    {Z X : Type _} {f : Z → X} {p : Z → Prop} :
    Definable f p ↔ FiberConstant f p := by
  constructor
  · rintro ⟨q, rfl⟩ z z' h
    change q (f z) ↔ q (f z')
    rw [h]
  · intro h
    refine ⟨fun x => ∃ z, f z = x ∧ p z, ?_⟩
    funext z
    apply propext
    constructor
    · intro hp
      exact ⟨z, rfl, hp⟩
    · rintro ⟨z', hf, hp⟩
      exact (h hf).mp hp

-- Boolean-valued packaging agrees with proposition-valued definability.
def BoolDefinable {Z X : Type _} (f : Z → X) (b : Z → Bool) : Prop :=
  ∃ q : X → Bool, b = q ∘ f

theorem boolDefinable_iff_fiberConstant
    {Z X : Type _} {f : Z → X} {b : Z → Bool} :
    BoolDefinable f b ↔ (∀ z z', f z = f z' → b z = b z') := by
  classical
  constructor
  · rintro ⟨q, rfl⟩ z z' h
    change q (f z) = q (f z')
    rw [h]
  · intro h
    refine ⟨fun x => decide (∃ z, f z = x ∧ b z = true), ?_⟩
    funext z
    change b z = decide (∃ z', f z' = f z ∧ b z' = true)
    have he : (∃ z', f z' = f z ∧ b z' = true) ↔ b z = true := by
      constructor
      · rintro ⟨z', hf, hb⟩
        exact (h z' z hf).symm.trans hb
      · intro hb
        exact ⟨z, rfl, hb⟩
    rw [propext he]
    cases b z <;> simp

theorem definable_true {Z X : Type _} (f : Z → X) :
    Definable f (fun _ => True) := ⟨fun _ => True, rfl⟩

theorem definable_false {Z X : Type _} (f : Z → X) :
    Definable f (fun _ => False) := ⟨fun _ => False, rfl⟩

theorem definable_and
    {Z X : Type _} {f : Z → X} {p r : Z → Prop}
    (hp : Definable f p) (hr : Definable f r) :
    Definable f (fun z => p z ∧ r z) := by
  rcases hp with ⟨qp, rfl⟩
  rcases hr with ⟨qr, rfl⟩
  refine ⟨fun x => qp x ∧ qr x, ?_⟩
  rfl

theorem definable_or
    {Z X : Type _} {f : Z → X} {p r : Z → Prop}
    (hp : Definable f p) (hr : Definable f r) :
    Definable f (fun z => p z ∨ r z) := by
  rcases hp with ⟨qp, rfl⟩
  rcases hr with ⟨qr, rfl⟩
  refine ⟨fun x => qp x ∨ qr x, ?_⟩
  rfl

theorem definable_not
    {Z X : Type _} {f : Z → X} {p : Z → Prop}
    (hp : Definable f p) :
    Definable f (fun z => ¬ p z) := by
  rcases hp with ⟨qp, rfl⟩
  refine ⟨fun x => ¬ qp x, ?_⟩
  rfl

theorem definable_xor
    {Z X : Type _} {f : Z → X} {p r : Z → Prop}
    (hp : Definable f p) (hr : Definable f r) :
    Definable f (fun z => (p z ∧ ¬r z) ∨ (r z ∧ ¬p z)) :=
  definable_or (definable_and hp (definable_not hr))
    (definable_and hr (definable_not hp))

-- Closure of expressible predicates does not assert persistence under
-- dynamics. This lemma explicitly requires a descended update to pull a
-- predicate back through that update.
theorem definable_preimage
    {Z X : Type _} {f : Z → X} {p : Z → Prop}
    {F : Z → Z} {g : X → X}
    (hp : Definable f p) (hF : ∀ z, f (F z) = g (f z)) :
    Definable f (p ∘ F) := by
  rcases hp with ⟨q, rfl⟩
  refine ⟨q ∘ g, ?_⟩
  funext z
  change q (f (F z)) = q (g (f z))
  rw [hF]

section Examples

def idBool (b : Bool) : Bool := b

def isTrue (b : Bool) : Prop := b = true

def isFalse (b : Bool) : Prop := b = false

theorem definable_isTrue : Definable idBool isTrue := by
  refine ⟨isTrue, ?_⟩
  rfl

theorem definable_isFalse : Definable idBool isFalse := by
  refine ⟨isFalse, ?_⟩
  rfl

example : Definable idBool (fun b => isTrue b ∨ isFalse b) := by
  exact definable_or definable_isTrue definable_isFalse

example : Definable idBool (fun b => ¬ isTrue b) := by
  exact definable_not definable_isTrue

example : Definable idBool (fun b => isTrue b ∧ ¬ isFalse b) := by
  exact definable_and definable_isTrue (definable_not definable_isFalse)

end Examples

end LogicClosure
