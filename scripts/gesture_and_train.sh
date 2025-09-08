#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/gesture_and_train.sh <gesture_name> "<description>" [extra train args...]
# Example:
#   scripts/gesture_and_train.sh ok_sign_2 \
#     "Index and thumb form a ring; other fingers relaxed; neutral wrist." \
#     --total-steps 20000 --hold 5 --max-steps 16 --eval-interval 200

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <gesture_name> \"<description>\" [extra train args...]" >&2
  exit 2
fi

GESTURE="$1"; shift
DESCRIPTION="$1"; shift
EXTRA_TRAIN_ARGS=("$@")

# -------- paths --------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="/proj/ciptmp/${USER}/envs/myosuite310/bin/python"   # pinned interpreter
GESTURES_DIR="${REPO_ROOT}/gestures"

# -------- load .env for OPENAI_API_KEY (if present) --------
if [[ -f "${REPO_ROOT}/.env" ]]; then
  # export lines of form KEY=VALUE (ignore comments/empty lines)
  while IFS='=' read -r key val; do
    [[ -z "${key}" || "${key}" =~ ^\# ]] && continue
    export "${key}"="${val}"
  done < <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "${REPO_ROOT}/.env" || true)
fi

# sanity check key
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set (put it in ${REPO_ROOT}/.env or export it)." >&2
  exit 3
fi

# -------- GPU / MuJoCo headless --------
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
ulimit -n 4096 || true

# -------- 1) Generate the gesture JSON via LLM (no training yet) --------
echo "[step] generating JSON for gesture '${GESTURE}' via OpenAI..."
"${PY}" "${REPO_ROOT}/scripts/gesture_to_train.py" \
  --name "${GESTURE}" \
  --prompt "${DESCRIPTION}" \
  --model "gpt-4o-mini" \
  --temperature 0.2 \
  --dry-run

SPEC_PATH="${GESTURES_DIR}/${GESTURE}.json"
if [[ ! -s "${SPEC_PATH}" ]]; then
  echo "ERROR: expected spec at ${SPEC_PATH} but it was not created." >&2
  exit 4
fi
echo "[ok] spec written → ${SPEC_PATH}"

# -------- 2) Prepare run folders (same style as your foreground launcher) --------
RUN_ID="$(date +%Y%m%d-%H%M%S)"
BASE_DIR="${REPO_ROOT}/runs/${GESTURE}/${RUN_ID}"
CKPT_DIR="${BASE_DIR}/checkpoints"

mkdir -p "${CKPT_DIR}"
mkdir -p "${REPO_ROOT}/runs/${GESTURE}"
ln -sfn "${BASE_DIR}" "${REPO_ROOT}/runs/${GESTURE}/latest"

SAVE_BASE="${CKPT_DIR}/checkpoint"
RESUME_ARGS=()
if [[ -f "${SAVE_BASE}.zip" ]]; then
  echo "[info] Resuming from checkpoint ${SAVE_BASE}.zip"
  RESUME_ARGS+=(--resume)
fi

echo
echo "[launch] gesture=${GESTURE}"
echo "  BASE_DIR: ${BASE_DIR}"
echo "  CKPT_DIR: ${CKPT_DIR}"
echo "  TRACES:   ${BASE_DIR}/traces"
echo

# -------- 3) Train in the foreground with pinned Python --------
exec "${PY}" -u "${REPO_ROOT}/scripts/train_rl_sac.py" \
  --env-id myoHandPoseFixed-v0 \
  --specs "${SPEC_PATH}" \
  --total-steps 20000 \
  --hold 5 \
  --max-steps 16 \
  --success 1000.0 \
  --logdir "${BASE_DIR}" \
  --save   "${SAVE_BASE}" \
  --eval-hold 20 \
  --eval-horizon 1 \
  --eval-interval 200 \
  --lam-effort 0.0 \
  --lam-smooth 0.0 \
  "${RESUME_ARGS[@]}" \
  "${EXTRA_TRAIN_ARGS[@]}"