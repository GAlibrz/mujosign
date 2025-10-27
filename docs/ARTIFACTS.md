# Artefact Layout

All solvers and RL evaluations call `src/mujosign/artifacts.write_run_artifacts`. This file describes the on-disk contract so that downstream tooling can inspect or post-process results confidently.

## Run Directory Structure

```
library/<gesture>/
├─ runs/<hash>/
│  ├─ activation.json
│  ├─ scores.json
│  ├─ pose_summary.json
│  ├─ gesture_spec.json
│  ├─ provenance.json
│  ├─ README.md
│  └─ thumb.png            # optional
├─ latest -> runs/<hash>   # symlink
├─ best_total -> runs/<hash>
├─ best_pose -> runs/<hash>
└─ index.json
```

- `<gesture>` matches the `gesture` field in the spec JSON.
- `<hash>` is derived from the `ProvenanceInputs` bundle (see below) so that identical inputs map to the same folder.

## File Contents

### `activation.json`
```json
{
  "env_id": "myoHandPoseFixed-v0",
  "muscles": ["abdPollLong", "..."],
  "activations": [0.42, 0.0, ...],
  "range": [0.0, 1.0],
  "units": "fraction",
  "time_horizon": "static"       // or "1xhold" for RL rollouts
}
```

### `scores.json`
Matches the dictionary returned by `mujosign.scoring.score`. Today only `pose_error` and `total` contain meaningful values; other fields are zero placeholders.

### `pose_summary.json`
```json
{
  "joint_angles_deg": { "thumb_MCP_flex": 59.8, "...": ... },
  "tip_positions_m": { "thumb_tip": [x, y, z], ... },
  "palm_normal_world": null
}
```

### `gesture_spec.json`
Copy of the spec dict used for the run (keeps provenance intact even if the source file changes later).

### `provenance.json`
```json
{
  "timestamp": "2025-09-08T19:40:12Z",
  "env_id": "myoHandPoseFixed-v0",
  "versions": {
    "mujosign": "0.0.1",
    "myosuite": "2.8.4",
    "mujoco": "3.1.2"
  },
  "solver": { "name": "fastpath", "seed": 0, "config": {...} },
  "spec_sha256": "…",
  "muscle_order_sha256": "…",
  "run_id": "<hash>"
}
```

### `README.md`
Human-readable summary created by `write_run_artifacts`, highlighting status (`ACCEPTED` vs `REJECTED`), total score, solver info, and metric breakdown.

### `thumb.png`
Optional thumbnail if the solver or RL evaluation managed to capture a frame (via Gym or dm_control). Absence of the file indicates rendering failed or was skipped.

### `index.json`
Aggregates all run metadata per gesture:
```json
{
  "runs": [
    {
      "id": "<hash>",
      "timestamp": "2025-09-08T19:40:12Z",
      "total": 412.3,
      "pose_error": 412.3,
      "accepted": false,
      "solver": "fastpath"
    }
  ],
  "aliases": {}
}
```

The helper also maintains three symlinks:
- `latest` → most recent run.
- `best_total` → run with lowest `total` score.
- `best_pose` → run with lowest `pose_error`.

## Provenance Hash

`ProvenanceInputs.hash()` builds a short SHA-256 digest of:
- Environment ID
- Gesture spec JSON (canonicalised)
- Solver configuration dict
- Version strings (`mujosign`, `myosuite`, `mujoco`)
- Muscle order list

This guarantees that identical inputs map to the same run directory, preventing duplicate artefacts for the same configuration.

## Usage Notes

- Artefact directories are immutable by design. If you need to annotate results, create new files inside `runs/<hash>/` rather than rewriting existing ones.
- When integrating with external reporting tools, rely on `index.json` to enumerate runs; the symlinks are conveniences, not authoritative records.
- SAC training may write multiple artefacts per evaluation interval. Use the timestamps inside `index.json` to reconstruct progression.
