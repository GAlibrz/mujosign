# Mujosign Runbook

This runbook captures the practical steps for authoring gestures, generating muscle activations, and curating artefacts with the current codebase.

## 1. Environment Preparation

1. Install dependencies (see `README.md`) and set a MuJoCo rendering backend, e.g.:
   ```bash
   export MUJOCO_GL=egl
   ```
2. Inspect the MyoSuite environment once so that actuator/joint name dumps exist:
   ```bash
   python scripts/inspect_names.py --env-id myoHandPoseFixed-v0
   ```
   This populates `assets/discovery/<env-id>/actuator_names.json`, which `load_muscle_order` requires.

## 2. Create or Update a Gesture Spec

### Option A – Manual authoring
1. Copy an existing file in `gestures/`.
2. Edit the `joints` dictionary using DOF names from `src/mujosign/utils/joint_names.py`.
3. Provide `target` (deg), `tolerance` (deg), and `weight` (float) per DOF.
4. Add a `notes` field to capture intent or references (optional).

### Option B – LLM-assisted authoring
1. Export `OPENAI_API_KEY` or place it in `.env`.
2. Run the helper:
   ```bash
   scripts/gesture_to_train.py \
     --name ok_sign_2 \
     --prompt "Index and thumb form a ring; other fingers relaxed; neutral wrist." \
     --dry-run
   ```
   Drop `--dry-run` to chain directly into training (see next section). The script validates the response and writes `gestures/<name>.json`.

### Validation

`schemas/gesture_spec.schema.json` is forward-looking and contains fields that the runtime does not yet use. Run validation when you want structural guarantees, but expect warnings if you omit optional-but-schema-required sections:
```bash
python scripts/validate_specs.py gestures/ok_sign_2.json
```

## 3. Generate Activations

### Static optimisation
```bash
python scripts/run_single.py \
  --env-id myoHandPoseFixed-v0 \
  --spec gestures/thumbs_up.json \
  --opt fastpath
```
This performs coordinate descent (`src/mujosign/solvers/fastpath.py`) and writes a complete artefact bundle under `library/thumbs_up/runs/<hash>/`.

### Reinforcement learning (SAC)
```bash
python scripts/train_rl_sac.py \
  --env-id myoHandPoseFixed-v0 \
  --specs gestures/v_sign.json \
  --total-steps 300000 \
  --logdir runs/v_sign/$(date +%Y%m%d-%H%M%S) \
  --save checkpoints/v_sign/latest
```
For long runs, prefer the managed launchers:
- `scripts/launch_train.sh <gesture> [extra args]` – nohup background run with log files under `/proj/.../mujosign_logs/`.
- `scripts/launch_train_foreground.sh <gesture>` – foreground run sharing the same defaults.
- `scripts/gesture_and_train.sh <gesture> "<description>"` – one-shot LLM spec generation followed by foreground SAC training.

Every evaluation step invokes `write_run_artifacts`, so the library stays up to date with the best run seen so far.

## 4. Inspect and Share Artefacts

- Render a thumbnail from the latest run:
  ```bash
  python scripts/render_pose.py \
    --env-id myoHandPoseFixed-v0 \
    --run-dir library/v_sign/latest \
    --out reports/v_sign.png
  ```
- Replay a trace (e.g., SAC rollout):
  ```bash
  python scripts/loop_replay.py \
    --env-id myoHandPoseFixed-v0 \
    --trace trace_20250829-043052_ep00696.npz
  ```
- Browse run metadata in `library/<gesture>/runs/<hash>/README.md`. Symlinks `latest`, `best_total`, and `best_pose` help you navigate active results quickly.

## Troubleshooting

- **`load_muscle_order` FileNotFoundError** – run `scripts/inspect_names.py` for the target `env-id`.
- **Blank or crashing renders** – confirm `MUJOCO_GL` is set to a backend supported by your driver (`egl`, `glfw`, etc.) and that you are inside the expected conda environment (see launcher scripts).
- **LLM returns invalid JSON** – re-run with a slightly tweaked prompt or lower `--temperature`; the helper already retries JSON parsing, but complex descriptions can still fail.
- **RL reward plateau** – adjust `--hold`, `--max-steps`, and `--total-steps`; consider widening tolerances in the spec if the policy never satisfies the success threshold.

## Useful One-liners

- Compare scores between runs:
  ```bash
  jq '.total' library/thumbs_up/runs/*/scores.json | sort -n
  ```
- Bulk render most recent runs:
  ```bash
  for g in gestures/*.json; do
      name=$(basename "${g%.json}")
      python scripts/render_pose.py --env-id myoHandPoseFixed-v0 \
        --run-dir "library/$name/latest" \
        --out "reports/${name}.png" || true
  done
  ```

