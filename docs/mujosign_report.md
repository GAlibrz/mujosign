# Mujosign Project Report

_Last updated: October 2025_

## 1. Problem Statement & Goals

Human hand gestures can be described declaratively (e.g., “thumbs up”, “OK sign”), but mapping those descriptions to physiologically plausible muscle activation vectors is non-trivial. Mujosign explores this bridge for Meta's MyoSuite hand models by:

1. Defining gestures in JSON with reproducible tolerances and weights.
2. Searching for muscle activation vectors through deterministic optimisation and reinforcement learning.
3. Capturing every run with rich provenance so results remain reproducible.
4. Providing tooling to render, replay, and iterate on gestures quickly.
5. Experimenting with language-model-assisted gesture authoring to accelerate spec creation.

## 2. Repository Snapshot

```
mujosign/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ docs/
│  ├─ ARCHITECTURE.md
│  ├─ RUNBOOK.md
│  ├─ SPEC_GESTURE_JSON.md
│  ├─ SPEC_SCORING.md
│  └─ mujosign_report.md  ← this file
├─ gestures/              # JSON gesture specs (manual + LLM-generated)
├─ library/               # Generated artefacts (runs per gesture)
├─ scripts/               # Training, rendering, gesture generation, utilities
├─ src/mujosign/
│  ├─ artifacts.py
│  ├─ rl/pose_env.py
│  ├─ scoring.py
│  ├─ sim_adapter.py
│  ├─ solvers/fastpath.py
│  └─ utils/joint_names.py
├─ configs/               # Example solver configs
├─ schemas/               # gesture_spec JSON schema
└─ runs/, reports/, trace_*  # Generated at runtime (not version-controlled)
```

Notable absences compared to early design drafts:
- No `tests/` directory yet (manual / ad-hoc validation only).
- No CMA-ES “synergy search” solver implementation; `fastpath` is the only static optimiser.
- No `specs.py` module; JSON is loaded ad-hoc.

## 3. Current Workflow

1. **Author Gesture Spec**
   - Manual edit or via `scripts/gesture_to_train.py` (LLM-assisted).
   - Specs are stored under `gestures/`.
2. **Inspect Environment**
   - `scripts/inspect_names.py` dumps joints/actuators/sites so that DOF mapping and muscle order are known.
3. **Simulation & Scoring**
   - `EnvAdapter` wraps Gymnasium's `myoHandPoseFixed-v0`, exposing joint angles in Mujosign's naming.
   - `scoring.py` currently computes pose error with per-joint tolerances.
4. **Search for Activations**
   - `scripts/run_single.py` runs coordinate descent (`fastpath`).
   - `scripts/train_rl_sac.py` trains a SAC agent in `PoseActivationEnv`, periodically evaluating and exporting the best activation.
   - Launcher scripts (`launch_train.sh`, `launch_train_foreground.sh`, `gesture_and_train.sh`) orchestrate environment setup and logging.
5. **Archive Artefacts**
   - `write_run_artifacts` saves activation vectors, scores, pose summaries, provenance (including hashes of spec and muscle order), a README card, and optional thumbnails under `library/<gesture>/runs/<hash>/`.
   - Symlinks (`latest`, `best_total`, `best_pose`) provide quick access to key runs per gesture.
6. **Inspect Results**
   - `render_pose.py`, `view_run.py`, and `loop_replay.py` render or replay results.

## 4. Component Notes

| Component | Status | Notes |
|-----------|--------|-------|
| Gesture schema (`schemas/gesture_spec.schema.json`) | Aspires to full feature set | Runtime currently uses only `gesture`, `joints`, `notes`. Other sections (relations, contacts, stability) are placeholders for future work. |
| Scoring (`src/mujosign/scoring.py`) | Minimal | Computes pose error; other metrics return `0.0`. |
| Env adapter (`src/mujosign/sim_adapter.py`) | Stable | Provides joint angles and fingertip positions; ready for relation/orientation scoring once implemented. |
| Static solver (`src/mujosign/solvers/fastpath.py`) | Stable | Deterministic coordinate descent with configurable step schedule. |
| RL env (`src/mujosign/rl/pose_env.py`) | Stable | Observations = spec features + current angles. Reward = negative total score minus optional effort/smoothness penalties. |
| SAC training (`scripts/train_rl_sac.py`) | Active | Supports resume, periodic evaluation, CSV logging, and artefact export. |
| LLM tooling (`scripts/gesture_to_train.py`, `scripts/gesture_and_train.sh`) | Experimental | Requires `openai` + `python-dotenv`. Relies on `OPENAI_API_KEY`. Generates spec JSON in repo before launching training. |
| Artefact writer (`src/mujosign/artifacts.py`) | Stable | Writes immutable run folders, updates `index.json`, and manages symlinks. |

## 5. Execution Cheatsheet

```bash
# Generate spec via LLM + run short SAC training
scripts/gesture_and_train.sh ok_sign_2 \
  "Index and thumb form a ring; other fingers relaxed; neutral wrist." \
  --total-steps 60000 --hold 10 --max-steps 16

# Static solver for an existing spec
python scripts/run_single.py --spec gestures/thumbs_up.json

# Long SAC run (nohup)
scripts/launch_train.sh v_sign --total-steps 1500000

# Render latest run thumbnail
python scripts/render_pose.py --run-dir library/v_sign/latest --out reports/v_sign.png
```

## 6. Status & Gaps

- ✅ **Gesture library** – multiple gestures (fist, thumbs up, v sign, OK variants, LLM-generated Italian gesture) available.
- ✅ **Artefact pipeline** – runs are archived with provenance + thumbnails (when render succeeds).
- ✅ **RL training** – SAC pipeline produces activations and integrates with artefact writer.
- ✅ **LLM integration** – one-shot script generates specs from natural language prompts.
- ⚠️ **Schema adherence** – existing gesture JSON files do not satisfy the full schema; validation is optional/informational.
- ⚠️ **Scoring depth** – relation/contact/orientation/effort terms are unimplemented.
- ⚠️ **Testing & CI** – no automated tests; manual inspection only.
- ⚠️ **Documentation debt** – schema field coverage, scoring roadmap, and RL tuning guidelines should be expanded further.

## 7. Next Steps

1. Implement additional scoring terms (relations, orientation, effort) and update specs/weights accordingly.
2. Align gesture JSON files with the full schema or relax the schema to match current usage.
3. Add automated tests (deterministic scoring, artefact writer invariants, env adapter integration).
4. Capture RL training heuristics (reward curves, hyperparameter sweeps) in a dedicated doc.
5. Explore alternative optimisers (CMA-ES or gradient-based) once scoring covers more terms.

## 8. References

- `docs/ARCHITECTURE.md` – component-level overview.
- `docs/RUNBOOK.md` – day-to-day operations guide.
- `docs/SPEC_GESTURE_JSON.md` – contract for gesture JSON.
- `docs/SPEC_SCORING.md` – scoring logic and roadmap.
- `README.md` – quickstart and repo map for newcomers.
