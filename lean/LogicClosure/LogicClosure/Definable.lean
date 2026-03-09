namespace LogicClosure

def fiberEq {Z X : Type _} (f : Z → X) (z z' : Z) : Prop := f z = f z'

def Definable {Z X : Type _} (f : Z → X) (p : Z → Prop) : Prop :=
  ∃ q : X → Prop, p = q ∘ f

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
