# Mujosign: Documentation-First Project Report

## 1. Problem Definition and Goals

### Motivation
The human hand can produce a rich vocabulary of gestures. Mapping these gestures to biologically plausible muscle activations is a non-trivial problem because the musculoskeletal system is redundant, nonlinear, and constrained by anatomy.

**Mujosign** is a documentation-first research project that builds an interface between gesture specifications (e.g., “V-sign”, “fist”, “thumbs up”) and the muscle activations in a physics-based simulator (**MyoSuite/MuJoCo**).

### Goals
1. Provide a specification format for gestures (JSON schemas).
2. Use optimizers and solvers to find muscle activation vectors that approximate those gestures.
3. Store all results in a reproducible gesture library with provenance metadata.
4. Enable rendering and replay of gestures for inspection and communication.
5. Establish a modular pipeline that can later include RL-based solvers or neural network optimizers.

---

## 2. Repository Structure (Authoritative)

```
mujosign/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ pyproject.toml
├─ conda-env.yml
├─ .env.example
│
├─ docs/
│  ├─ ARCHITECTURE.md
│  ├─ SPEC_GESTURE_JSON.md
│  ├─ SPEC_SCORING.md
│  ├─ RUNBOOK.md
│  ├─ TESTING.md
│  ├─ DATA_MODEL.md
│  ├─ LOGGING_PROVENANCE.md
│  ├─ CONTRIBUTING.md
│  ├─ ROADMAP.md
│  ├─ GLOSSARY.md
│  └─ ADR/
│     └─ ADR-0001-gesture-schema.md
│
├─ gestures/
│  ├─ fist.json
│  ├─ thumbs_up.json
│  ├─ v_sign.json
│  └─ pinch.json
│
├─ configs/
│  ├─ camera.default.yaml
│  ├─ scoring.v1.yaml
│  ├─ solver.fastpath.yaml
│  ├─ solver.synergy.yaml
│  └─ batch.default.yaml
│
├─ scripts/
│  ├─ run_single.py
│  ├─ run_batch.py
│  ├─ validate_specs.py
│  ├─ summarize_library.py
│  └─ render_pose.py
│
├─ src/mujosign/
│  ├─ sim_adapter.py
│  ├─ scoring.py
│  ├─ specs.py
│  ├─ solvers/
│  │  ├─ fastpath.py
│  │  ├─ synergy_search.py
│  │  └─ utils.py
│  ├─ artifacts.py
│  ├─ rendering.py
│  └─ utils/
│     ├─ joint_names.py
│     ├─ rom_tables.py
│     └─ logging_utils.py
│
├─ library/
│  └─ <gesture-name>/
│     ├─ activation.json
│     ├─ pose_summary.json
│     ├─ scores.json
│     ├─ provenance.json
│     ├─ gesture_spec.json
│     └─ thumb.png
│
├─ reports/
│  ├─ library_summary.md
│  └─ metrics.csv
│
├─ schemas/
│  ├─ gesture_spec.schema.json
│  └─ gesture_card.schema.json
│
├─ tests/
│  ├─ test_scoring_determinism.py
│  ├─ test_rom_guards.py
│  ├─ test_contact_rules.py
│  ├─ test_render_consistency.py
│  └─ test_spec_validation.py
│
└─ tools/
   ├─ precommit-config.yaml
   └─ make_thumbnails.sh
```

**Rules & Conventions**
- Docs > Code: docs are authoritative.
- Deterministic scoring.
- ROM & tendon constraints always enforced.
- Immutable artifacts per provenance hash.
- Specs validated against schema.

---

## 3. Workflow Overview

1. **Gesture spec (JSON)** → defines joint targets, tolerances, weights.
2. **Validation** → `validate_specs.py` checks against schema.
3. **Inspection** → `inspect_names.py` enumerates joints, muscles, sites.
4. **Optimization** → `run_single.py` runs solver (fastpath, synergy, etc.).
5. **Artifacts** → JSON + PNG stored in `library/gesture/runs/hash/`.
6. **Replay/Render** → `render_pose.py` produces thumbnails; `view_run.py` interactive viewer.

---

## 4. Component Documentation

### `inspect_names.py`
Extracts joint, actuator, and site names from a MyoSuite env and writes them to CSV/JSON for reference.

### `joint_names.py`
Defines `DOF_MAP`, `TIP_SITES`, and stable muscle order, used by adapters and solvers.

### `run_single.py`
Main runner: takes `--spec` gesture JSON, applies solver, scores pose, writes artifacts.

### `fastpath.py`
Implements coordinate descent optimization to minimize pose error.

### `artifacts.py`
Writes activation.json, pose_summary.json, provenance.json, thumb.png for reproducibility.

### `render_pose.py`
Replays a saved run and re-renders an image of the gesture.

### `view_run.py`
Interactive GUI viewer (dm_control.viewer) to see gestures in 3D.

### `validate_specs.py`
Validates all gestures against `schemas/gesture_spec.schema.json`.

... (and so on, with explanations for configs, schemas, tests)

---

## 5. Execution Instructions

### Setup
```bash
conda create -n myosuite python=3.8
conda activate myosuite
pip install -e .
export MUJOCO_GL=glfw
```

### Validate specs
```bash
python scripts/validate_specs.py gestures/
```

### Run optimization
```bash
python scripts/run_single.py --spec gestures/v_sign.json --opt fastpath --opt-config configs/solver.fastpath.yaml
```

### Re-render run
```bash
python scripts/render_pose.py --env-id myoHandPoseFixed-v0 --run-dir library/v_sign/runs/<hash> --out reports/replay.png
```

### Interactive view
```bash
python scripts/view_run.py --env-id myoHandPoseFixed-v0 --run-dir library/v_sign/runs/<hash>
```

---

## 6. Evolution Path

- Started with documentation-first repo skeleton.
- Added gesture JSON specs + schema validation.
- Inspected sim (joints, actuators, sites).
- Defined DOF maps and stable muscle order.
- Implemented runner (`run_single.py`) with fastpath solver.
- Built artifacts writer with provenance metadata.
- Added re-render (`render_pose.py`) and interactive viewer (`view_run.py`).
- Next steps: improve scoring, add RL-based solvers, extend relation encoding.

---

## 7. Next Steps

- Implement RL agent to optimize activations across episodes.
- Integrate relation-based scoring.
- Improve rendering quality and camera control.
- Add batch runner + metrics dashboards.
