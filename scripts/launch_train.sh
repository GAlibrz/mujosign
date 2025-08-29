#!/usr/bin/env bash
set -euo pipefail

# -------- args --------
GESTURE="${1:?usage: $0 <gesture> [extra-args...]}"
shift || true
EXTRA_ARGS=("$@")

# -------- conda + env --------
# Make sure this conda is first on PATH
export PATH="/proj/ciptmp/${USER}/conda/bin:${PATH}"

# Load conda into current shell (if available)
if [[ -f "/proj/ciptmp/${USER}/conda/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "/proj/ciptmp/${USER}/conda/etc/profile.d/conda.sh"
else
  echo "WARNING: conda.sh not found; continuing, but 'conda activate' may fail." >&2
fi

# Clear any hashed command paths
hash -r || true

# Activate env by absolute path (prefix)
ENV_PREFIX="/proj/ciptmp/${USER}/envs/myosuite310"
conda activate "${ENV_PREFIX}"

echo "Python: $(python -V)"
echo "Which python: $(which python)"

# -------- rendering / gpu --------
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

# (optional) avoid too many open files issues in long runs
ulimit -n 4096 || true

# -------- folders --------
RUN_ID="$(date +%Y%m%d-%H%M%S)"
BASE_DIR="$PWD/runs/$GESTURE/$RUN_ID"   # train --logdir (traces go here)
CKPT_DIR="$BASE_DIR/checkpoints"
LOG_DIR="/proj/ciptmp/${USER}/mujosign_logs/$GESTURE/$RUN_ID"  # noisy stdout/err

# refuse to overwrite anything that already exists
for d in "$BASE_DIR" "$CKPT_DIR" "$LOG_DIR"; do
  if [[ -e "$d" ]]; then
    echo "Refusing to overwrite existing path: $d" >&2
    exit 1
  fi
done

mkdir -p "$CKPT_DIR" "$LOG_DIR"

# Maintain a 'latest' symlink per-gesture inside repo
mkdir -p "runs/$GESTURE"
ln -sfn "$BASE_DIR" "runs/$GESTURE/latest"

# -------- resume logic --------
SAVE_BASE="$CKPT_DIR/checkpoint"
RESUME_ARGS=()
if [[ -f "${SAVE_BASE}.zip" ]]; then
  echo "[info] Found existing checkpoint: ${SAVE_BASE}.zip — will resume."
  RESUME_ARGS+=(--resume)
fi

# -------- launch --------
echo "[launch] gesture=${GESTURE}"
echo "  BASE_DIR:    $BASE_DIR"
echo "  CKPT_DIR:    $CKPT_DIR"
echo "  LOG_DIR:     $LOG_DIR"
echo "  SAVE_BASE:   $SAVE_BASE"
echo "  TRACES:      $BASE_DIR/traces"
echo "  LATEST LINK: runs/$GESTURE/latest -> $BASE_DIR"

PYTHONUNBUFFERED=1 nohup python -u scripts/train_rl_sac.py \
  --env-id myoHandPoseFixed-v0 \
  --specs "gestures/${GESTURE}.json" \
  --total-steps 1500000 \
  --hold 19 \
  --max-steps 48 \
  --success 1000.0 \
  --logdir "$BASE_DIR" \
  --save   "$SAVE_BASE" \
  --eval-hold 19 \
  --eval-horizon 1 \
  --eval-interval 1000 \
  "${RESUME_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  1>"$LOG_DIR/train.stdout.log" \
  2>"$LOG_DIR/train.stderr.log" &

PID=$!
echo "Started PID=$PID"
echo
echo "Tail logs:"
echo "  tail -f $LOG_DIR/train.stdout.log"
echo "  tail -f $LOG_DIR/train.stderr.log"
echo
echo "Quick status:"
echo "  ps -fp $PID || true"
echo
echo "Progress CSV:"
echo "  watch -n 5 'tail -n 5 $BASE_DIR/progress.csv'"