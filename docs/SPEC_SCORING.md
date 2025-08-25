# Scoring Specification

## Inputs from Simulation
- Joint angles (deg), velocities
- Fingertip positions (mm)
- Palm pose / orientation
- Contacts (boolean pairs)
- Muscle activations
- Tendon strain/forces

## Terms
- **Pose Error**: per joint, tolerance-normalized squared error.
- **Relation Error**: derived metrics (abduction, tip distance).
- **Contact Error**: penalties for missing required or hitting forbidden contacts.
- **Orientation Error**: wrist angles and palm normal alignment.
- **Effort**: mean ∑(activation²).
- **Smoothness**: mean squared change in activations.
- **Hard Penalties**: ROM violation, tendon violation, instability.

## Total Score
```
Score = Σ W_i * E_i + hard_penalties
```
Weights `W_i` taken from gesture JSON.

## Acceptance Criteria
- No ROM, tendon, or stability violations.
- Pose + relation error ≤ τ (gesture-specific).
- Stability hold window satisfied.
