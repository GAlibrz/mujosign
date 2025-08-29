#!/usr/bin/env bash
set -euo pipefail

GESTURE="${1:?usage: $0 <gesture> [extra-args...]}"
shift || true
EXTRA_ARGS=("$@")

# 1) Unique run folder inside repo (safe)
RUN_ID="$(date +%Y%m%d-%H%M%S)"
BASE_DIR="$PWD/runs/$GESTURE/$RUN_ID"   # <- use this as --logdir so traces go here
CKPT_DIR="$BASE_DIR/checkpoints"
TRACE_DIR="$BASE_DIR/traces"            # <- train script will create <logdir>/traces

# 2) Ephemeral console logs in ciptmp (separate from --logdir)
LOG_DIR="/proj/ciptmp/${USER}/mujosign_logs/$GESTURE/$RUN_ID"

# 3) Create dirs; refuse to overwrite
for d in "$BASE_DIR" "$CKPT_DIR" "$LOG_DIR"; do
  if [[ -e "$d" ]]; then
    echo "Refusing to overwrite existing path: $d" >&2
    exit 1
  fi
done
mkdir -p "$CKPT_DIR" "$LOG_DIR"

# 4) Symlink a 'latest' pointer per gesture
ln -sfn "$BASE_DIR" "runs/$GESTURE/latest"

# 5) Launch training
#    NOTE: --logdir points to $BASE_DIR so traces will be written to $BASE_DIR/traces
PYTHONUNBUFFERED=1 nohup python3 -u scripts/train_rl_sac.py \
  --env-id myoHandPoseFixed-v0 \
  --specs "gestures/${GESTURE}.json" \
  --total-steps 600000 \
  --hold 20 \
  --max-steps 16 \
  --success 1000.0 \
  --logdir "$BASE_DIR" \
  --save   "$CKPT_DIR/checkpoint" \
  --eval-hold 20 \
  --eval-horizon 1 \
  --eval-interval 10000 \
  "${EXTRA_ARGS[@]}" \
  1>"$LOG_DIR/train.stdout.log" \
  2>"$LOG_DIR/train.stderr.log" &

PID=$!
echo "Started PID=$PID"
echo "RUN_ID=$RUN_ID"
echo "BASE_DIR=$BASE_DIR"
echo "  checkpoints: $CKPT_DIR"
echo "  traces:      $BASE_DIR/traces"
echo "  logs:        $LOG_DIR"
echo "Tail logs:"
echo "  tail -f $LOG_DIR/train.stdout.log"