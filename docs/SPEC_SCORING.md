# Scoring Specification

`src/mujosign/scoring.py` currently implements the **pose error** term and returns placeholders for the other buckets so that the API shape stays stable as we grow the metric set. The function is deterministic and side-effect free: supplying the same spec and joint-angle dictionary always yields the same numeric breakdown.

## Inputs from Simulation

- Joint angles in degrees keyed by Mujosign DOF names (`EnvAdapter.get_joint_angles_deg()`).
- Gesture spec dictionary (typically loaded from JSON).

Future terms will also require fingertip positions, contact information, muscle activations, and pose stability windows; the adapter already exposes enough data for those extensions.

## Pose Error (implemented)

For every DOF in `spec["joints"]`:

```
target      = rule["target"]
tolerance   = rule["tolerance"]
weight      = rule["weight"]
value       = current_deg.get(dof, 0.0)
excess      = max(0, |value - target| - tolerance)
pose_error += weight * excess²
```

The total score reported today is the pose error alone. All other fields in the returned dictionary (`relation_error`, `contact_error`, `effort`, etc.) are set to `0.0` so downstream consumers do not break when we add real implementations.

## Planned Extensions

The JSON schema already reserves sections for richer constraints:

- **Relation error** – tip distances, finger abduction angles, or multi-joint expressions derived from `TIP_SITES`.
- **Contact error** – required/forbidden contact pairs between geometric sites.
- **Orientation error** – wrist pitch/yaw and palm-frame orientation targets.
- **Effort / smoothness** – quadratic penalties on activations and activation deltas (available in RL loops).
- **ROM / tendon / stability violations** – hard penalties once the safety monitors are wired in.

When these terms are implemented, the scoring contract will remain a dictionary with the same keys; additional metrics will simply transition from zero placeholders to computed values.
