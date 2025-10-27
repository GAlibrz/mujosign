# Mujosign

Mujosign is a tooling stack for discovering physiologically plausible muscle activation vectors that reproduce static hand gestures in Meta's [MyoSuite](https://github.com/facebookresearch/myosuite) MuJoCo environments. The project formalises gesture targets, runs optimisation or reinforcement learning to satisfy them, and snapshots every interesting result with provenance so that runs are reproducible.

## Overview

- **Gesture specs** – Each pose is described by JSON files under `gestures/` (see `schemas/gesture_spec.schema.json`). Specs constrain joint angles, tolerances, and weights by the gesture's name.
- **LLM-assisted authoring** – `scripts/gesture_to_train.py` and `scripts/gesture_and_train.sh` can turn a natural-language description into a validated spec using OpenAI's API before launching training.
- **Simulation adapters** – `src/mujosign/sim_adapter.py` wraps Gymnasium/MyoSuite environments and exposes joint angles (degrees) and fingertip positions in Mujosign's naming scheme.
- **Scoring & optimisation** – `src/mujosign/scoring.py` measures how well the current pose matches a spec. `src/mujosign/solvers/fastpath.py` provides a coordinate-descent optimiser for static searches.
- **RL training loop** – `scripts/train_rl_sac.py` trains a SAC policy in `PoseActivationEnv` (`src/mujosign/rl/pose_env.py`), periodically evaluating, checkpointing, and exporting artefacts.
- **Artefact archive** – `src/mujosign/artifacts.py` writes library entries under `library/<gesture>/runs/<hash>/` containing activations, scores, summaries, and optional thumbnails plus best/latest symlinks.

## Requirements

- Python 3.10+ with CUDA-capable drivers if you plan to train RL locally.
- MuJoCo 3.x and MyoSuite 2.x (pulled in via `requirements.txt`).
- Optional: `openai` and `python-dotenv` if you want LLM-based gesture creation (set `OPENAI_API_KEY` via `.env` or the shell).

Install dependencies:

```bash
pip install -r requirements.txt
# Optional for LLM tooling:
pip install openai python-dotenv
```

If you have not yet inspected the MyoSuite environment metadata, generate joint/actuator name dumps once:

```bash
python scripts/inspect_names.py --env-id myoHandPoseFixed-v0
```

This writes discovery files under `assets/discovery/<env-id>/` which are consumed by `load_muscle_order`.

## Quickstart

1. **Run a fast static search**
   ```bash
   python scripts/run_single.py --env-id myoHandPoseFixed-v0 \
     --spec gestures/thumbs_up.json \
     --opt fastpath \
     --opt-config configs/solver.fastpath.yaml
   ```
   The script prints the resulting activation vector and drops a new run into `library/thumbs_up/runs/…`.

2. **Train with SAC**
   ```bash
   python scripts/train_rl_sac.py \
     --env-id myoHandPoseFixed-v0 \
     --specs gestures/v_sign.json \
     --total-steps 300000 \
     --logdir runs/v_sign/$(date +%Y%m%d-%H%M%S) \
     --save checkpoints/v_sign/latest
   ```
   Training checkpoints and evaluation artefacts are created automatically; CSV logs land in the chosen `--logdir`.

3. **Render or replay results**
   ```bash
   python scripts/render_pose.py --env-id myoHandPoseFixed-v0 \
     --run-dir library/v_sign/latest \
     --out reports/v_sign.png
   python scripts/loop_replay.py --env-id myoHandPoseFixed-v0 \
     --trace trace_20250829-043052_ep00696.npz
   ```

## Gesture Creation with LLMs

The new gesture workflow (requires `OPENAI_API_KEY`):

```bash
export OPENAI_API_KEY=sk-...
scripts/gesture_and_train.sh ok_sign_2 \
  "Index and thumb form a ring, other fingers relaxed, neutral wrist." \
  --total-steps 60000 --hold 10 --max-steps 16
```

Steps performed:

1. `scripts/gesture_to_train.py` prompts OpenAI (`gpt-4o-mini` by default) to produce a spec compliant with Mujosign's DOF map.
2. The JSON spec is validated and saved to `gestures/<gesture>.json`.
3. Training launches immediately via `scripts/train_rl_sac.py` with defaults that can be overridden through additional flags.

Use `--dry-run` to generate specs without kicking off training.

## Launchers & Automation

- `scripts/gesture_and_train.sh` – end-to-end generator + trainer using a pinned Python interpreter.
- `scripts/launch_train.sh` – nohup wrapper for long SAC runs. Configures conda envs and logs under `/proj/.../mujosign_logs/`.
- `scripts/launch_train_foreground.sh` – interactive foreground variant (fix the trailing line continuation if you customise it).
- `scripts/view_run.py`, `scripts/view_trace_dmcontrol.py`, and `scripts/view_traces_playlist.py` – quick inspection utilities for saved runs and traces.

## Artefact Library

Each optimisation or RL evaluation writes a self-contained directory:

```
library/<gesture>/
├─ runs/<short_hash>/
│  ├─ activation.json      # muscle order + activation vector
│  ├─ scores.json          # scoring breakdown (pose_error, total, ...)
│  ├─ pose_summary.json    # joint angles, tip positions, palm normal placeholder
│  ├─ gesture_spec.json    # spec used for the run
│  ├─ provenance.json      # env, solver config, version hashes, timestamp
│  ├─ README.md            # human-readable summary
│  └─ thumb.png            # optional thumbnail if rendering succeeded
├─ latest -> runs/<…>      # symlink to most recent run
├─ best_total -> runs/<…>  # best total score symlink
└─ index.json              # summary list of all runs
```

These artefacts enable reproducibility (hashes of spec and muscle order) and make it easy to inspect progress per gesture.

## Repository Layout

```
mujosign/
├─ gestures/                 # JSON gesture specs (LLM or manual)
├─ library/                  # Generated artefacts (per gesture/run)
├─ scripts/                  # Tooling: training, rendering, gesture generation
├─ src/mujosign/
│  ├─ artifacts.py           # Library writer utilities
│  ├─ rl/pose_env.py         # Gymnasium env for RL
│  ├─ scoring.py             # Pose scoring helpers
│  ├─ sim_adapter.py         # MyoSuite adapter
│  ├─ solvers/fastpath.py    # Static optimiser
│  └─ utils/joint_names.py   # DOF definitions and loaders
├─ configs/                  # Solver configs, example YAML
├─ docs/                     # High-level specs & architecture notes
└─ README.md
```

Explore `docs/` for deeper design rationale (`SPEC_GESTURE_JSON.md`, `SPEC_SCORING.md`, `ARCHITECTURE.md`), and check `configs/` / `scripts/` for more examples.

