import LogicClosure.Definable
import LogicClosure.QuotientDynamics

namespace LogicClosure

-- Exact noise-free register updates matching the Python laboratories.
def notUpdate (p : Bool × Bool) : Bool × Bool := (p.1, !p.1)

def cnotUpdate (p : Bool × Bool) : Bool × Bool := (p.1, Bool.xor p.1 p.2)

def andUpdate (p : (Bool × Bool) × Bool) : (Bool × Bool) × Bool :=
  (p.1, p.1.1 && p.1.2)

theorem notUpdate_input_respects :
    Respects (kernelSetoid (fun p : Bool × Bool => p.1)) notUpdate := by
  intro a b h
  exact h

theorem andUpdate_input_respects :
    Respects (kernelSetoid (fun p : (Bool × Bool) × Bool => p.1)) andUpdate := by
  intro a b h
  exact h

theorem cnotUpdate_involutive (p : Bool × Bool) :
    cnotUpdate (cnotUpdate p) = p := by
  rcases p with ⟨a, b⟩
  cases a <;> cases b <;> rfl

theorem cnotUpdate_injective :
    ∀ a b, cnotUpdate a = cnotUpdate b → a = b := by
  intro a b h
  have hh := congrArg cnotUpdate h
  simpa only [cnotUpdate_involutive] using hh

-- Erasing the control prevents a one-step state update on the target alone.
theorem cnotUpdate_target_not_respects :
    ¬ Respects (kernelSetoid (fun p : Bool × Bool => p.2)) cnotUpdate := by
  intro h
  have bad := h (a := (false, false)) (b := (true, false)) rfl
  change false = true at bad
  cases bad

theorem cnotUpdate_target_no_macroMap :
    ¬ ∃ g : Bool → Bool, ∀ p, (cnotUpdate p).2 = g p.2 := by
  rintro ⟨g, hg⟩
  have h0 := hg (false, false)
  have h1 := hg (true, false)
  have bad : false = true := h0.trans h1.symm
  cases bad

-- Horizon matters: after two ideal ticks the incomplete target lens closes.
theorem cnotUpdate_two_ticks_respects :
    Respects (kernelSetoid (fun p : Bool × Bool => p.2))
      (cnotUpdate ∘ cnotUpdate) := by
  intro a b h
  change (cnotUpdate (cnotUpdate a)).2 = (cnotUpdate (cnotUpdate b)).2
  simpa only [cnotUpdate_involutive] using h

-- A perfect NOT input/output channel does not imply output-state closure.
theorem notUpdate_output_not_respects :
    ¬ Respects (kernelSetoid (fun p : Bool × Bool => p.2)) notUpdate := by
  intro h
  have bad := h (a := (false, false)) (b := (true, false)) rfl
  change true = false at bad
  cases bad

end LogicClosure
