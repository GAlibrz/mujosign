# Mujosign Architecture

## Components
- **Spec Author** → defines gestures as JSON specs in `gestures/`.
- **Sim Adapter** → wraps MyoSuite hand env, exposes state (joint angles, fingertip positions, contacts).
- **Scorer** → deterministic objective, compares sim state to gesture spec.
- **Search Controller** → solver (fast path IK→static opt, fallback synergy search).
- **Artifact Writer** → saves outputs (activation vector, scores, provenance, thumbnail).
- **Batch Orchestrator** → iterates gestures and aggregates results.

## Data Flow
```
GestureSpec (JSON) → Sim Adapter → Scorer
          → Search Controller ↔ Simulation
          → GestureCard (artifacts) → library/
```

## Invariants
- Scorer must be pure/deterministic given state+spec.
- ROM, tendon, stability constraints are enforced always.
- Artifacts include provenance (versions, seed, config).
- Rendering uses fixed camera config for consistency.
