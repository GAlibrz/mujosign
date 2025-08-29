#!/usr/bin/env bash
set -euo pipefail

# -------- args --------
GESTURE="${1:?usage: $0 <gesture> [extra-args...]}"
shift || true
EXTRA_ARGS=("$@")

# -------- conda + env --------
export PATH="/proj/ciptmp/${USER}/conda/bin:${PATH}"
if [[ -f "/proj/ciptmp/${USER}/conda/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "/proj/ciptmp/${USER}/conda/etc/profile.d/conda.sh"
fi
hash -r || true
conda activate "/proj/ciptmp/${USER}/envs/myosuite310"

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
ulimit -n 4096 || true

# -------- folders --------
RUN_ID="$(date +%Y%m%d-%H%M%S)"
BASE_DIR="$PWD/runs/$GESTURE/$RUN_ID"
CKPT_DIR="$BASE_DIR/checkpoints"

mkdir -p "$CKPT_DIR"
mkdir -p "runs/$GESTURE"
ln -sfn "$BASE_DIR" "runs/$GESTURE/latest"

SAVE_BASE="$CKPT_DIR/checkpoint"
RESUME_ARGS=()
if [[ -f "${SAVE_BASE}.zip" ]]; then
  echo "[info] Resuming from checkpoint ${SAVE_BASE}.zip"
  RESUME_ARGS+=(--resume)
fi

# -------- launch (FOREGROUND) --------
echo "[launch] gesture=${GESTURE}"
echo "  BASE_DIR: $BASE_DIR"
echo "  CKPT_DIR: $CKPT_DIR"
echo "  TRACES:   $BASE_DIR/traces"
echo

python -u scripts/train_rl_sac.py \
  --env-id myoHandPoseFixed-v0 \
  --specs "gestures/${GESTURE}.json" \
  --total-steps 20000 \
  --hold 5 \
  --max-steps 16 \
  --success 1000.0 \
  --logdir "$BASE_DIR" \
  --save   "$SAVE_BASE" \
  --eval-hold 20 \
  --eval-horizon 1 \
  --eval-interval 200 \
  --lam-effort 0.0 \
  --lam-smooth 0.0
  "${RESUME_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"