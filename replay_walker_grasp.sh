#!/usr/bin/env bash
# Replay a Walker S2 grasp episode recorded in local LeRobot format.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_ROOT="${WALKER_GRASP_DATASET_ROOT:-${SCRIPT_DIR}/recordings/walker_s2_grasp_train}"
REPO_ID="${WALKER_GRASP_REPO_ID:-walker_s2_grasp}"
EPISODE="${WALKER_GRASP_EPISODE:-0}"

exec "${SCRIPT_DIR}/teleop_walker_grasp.sh" \
  --replay-dataset-root "$DATASET_ROOT" \
  --replay-repo-id "$REPO_ID" \
  --replay-episode "$EPISODE" \
  "$@"
