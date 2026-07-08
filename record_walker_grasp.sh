#!/usr/bin/env bash
# Record one Walker S2 grasp episode in local LeRobot format.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_ROOT="${WALKER_GRASP_DATASET_ROOT:-${SCRIPT_DIR}/recordings/walker_s2_grasp_train}"
REPO_ID="${WALKER_GRASP_REPO_ID:-walker_s2_grasp}"
TASK="${WALKER_GRASP_TASK:-pick the block and place it in the tray}"

exec "${SCRIPT_DIR}/teleop_walker_grasp.sh" \
  --record-dataset-root "$DATASET_ROOT" \
  --record-repo-id "$REPO_ID" \
  --record-task "$TASK" \
  --record-start task \
  "$@"
