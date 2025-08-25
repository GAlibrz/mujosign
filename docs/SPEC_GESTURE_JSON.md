# GestureSpec JSON Schema

## Schema v1
```json
{
  "gesture": "string",
  "handedness": "right | left",
  "model": "myoHand",
  "joints": {
    "<dof_name>": { "target": 0, "tolerance": 10, "weight": 1.0, "mode": "value|min|max|range" }
  },
  "relations": {
    "index_middle_abduction_deg": { "min": 15, "weight": 1.0 },
    "thumb_index_tip_distance_mm": { "target": 0, "tolerance": 3, "weight": 1.0 }
  },
  "contacts": { "require": ["thumb_pad:index_pad"], "forbid": ["index_tip:palm"] },
  "orientation": {
    "wrist_pitch_deg": { "target": 0, "tolerance": 15, "weight": 0.5 },
    "wrist_yaw_deg":   { "target": 0, "tolerance": 15, "weight": 0.5 },
    "palm_normal_world": "forward|up|any",
    "palm_normal_tolerance_deg": 20
  },
  "stability": { "hold_steps": 50, "max_joint_drift_deg": 5, "max_tip_drift_mm": 5 },
  "weights": {
    "pose_error": 1.0, "relation_error": 1.0, "contact_error": 1.0,
    "orientation_error": 0.5, "effort": 0.01, "smoothness": 0.005,
    "rom_violation": 1000.0, "tendon_violation": 1000.0, "stability_violation": 100.0
  },
  "notes": "free text"
}
```

## Notes
- Use tolerances instead of exact values.
- DOF naming must be mapped to MyoSuite’s hand model (to be documented in `src/mujosign/utils/joint_names.py`).
- Relations capture multi-joint properties (e.g. finger splay).
