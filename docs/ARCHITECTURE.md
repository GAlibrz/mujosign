# Mujosign Architecture

Mujosign discovers muscle activation vectors that recreate static hand poses in MyoSuite. The system is intentionally modular so that new gesture-authoring approaches or solving strategies can be plugged in without rewriting the core.

## Component Map

- **Gesture authoring**
  - Manual JSON specs under `gestures/`.
  - `scripts/gesture_to_train.py` can translate a natural-language description into a JSON spec via OpenAI; `scripts/gesture_and_train.sh` wraps generation + training.
  - Specs are validated against `schemas/gesture_spec.schema.json` (currently the schema is broader than the subset consumed by the code).
- **Simulation adapter**
  - `src/mujosign/sim_adapter.EnvAdapter` wraps a Gymnasium/MyoSuite environment.
  - Exposes joint angles (degrees) keyed by Mujosign DOF names and fingertip world positions.
- **Scoring**
  - `src/mujosign/scoring.py` evaluates how well the current pose matches the spec. Right now only the pose error term is implemented; other buckets are stubs for future expansion.
- **Solvers**
  - `src/mujosign/solvers/fastpath.optimize` runs a deterministic coordinate-descent search over muscle activations; used by `scripts/run_single.py`.
  - `src/mujosign/rl/pose_env.PoseActivationEnv` turns the problem into an RL task. `scripts/train_rl_sac.py` trains a SAC agent, periodically evaluating and exporting artefacts.
- **Artefact writer**
  - `src/mujosign/artifacts.write_run_artifacts` snapshots each evaluated activation: activation.json, scores.json, pose_summary.json, provenance.json, README.md, optional thumbnail, and index management.
- **Launchers & tooling**
  - Batch/foreground launch scripts manage env activation, logging, and run folders (`scripts/launch_train.sh`, `scripts/launch_train_foreground.sh`, `scripts/gesture_and_train.sh`).
  - Viewer/render utilities (`scripts/render_pose.py`, `scripts/view_run.py`, `scripts/loop_replay.py`) provide inspection.

## Data Flow

```
Gesture description ─┐
                     ├─> GestureSpec JSON ──┐
Manual authoring ────┘                      │
                                            ▼
                                      EnvAdapter (MyoSuite)
                                            │
                                  joint angles + tips
                                            ▼
    ┌─────────────── static solver (fastpath) ───────────────┐
    │                                                       │
    │      activation guess ──> rollout ──> scoring ──┐      │
    │                                                ▼      │
    └────────────── RL agent (PoseActivationEnv + SAC) ─────┘
                                            │
                                            ▼
                                   write_run_artifacts
                                            │
                                            ▼
                                   library/<gesture>/runs/<hash>
```

Both the static solver and the RL policy rely on the same scoring function and artefact writer, keeping downstream analytics consistent.

## Invariants

- Scoring must be deterministic for a given spec + simulator state.
- Artefact directories are immutable once written; symlinks (`latest`, `best_total`, `best_pose`) provide the mutable handles.
- Provenance files include hashes of the gesture spec and muscle order so that results remain reproducible even if JSON files are edited later.
- Launch scripts always establish a known environment (conda env + `MUJOCO_GL=egl`) before touching the simulator.
