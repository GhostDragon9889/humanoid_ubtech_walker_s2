#!/usr/bin/env bash
# Roll out a trained Walker S2 LeRobot policy in Isaac Sim.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -n "${ISAAC_SIM_PYTHON:-}" ]]; then
  ISAAC_PYTHON="$ISAAC_SIM_PYTHON"
elif [[ -x /isaac-sim/python.sh ]]; then
  ISAAC_PYTHON=/isaac-sim/python.sh
elif [[ -x "${HOME}/isaacsim/python.sh" ]]; then
  ISAAC_PYTHON="${HOME}/isaacsim/python.sh"
else
  echo "Isaac Sim python.sh was not found." >&2
  echo "Set ISAAC_SIM_PYTHON=/path/to/isaacsim/python.sh." >&2
  exit 1
fi

DEFAULT_URDF="${SCRIPT_DIR}/assets/resources/walker_s2_description_hand3_v1_left_hand3_v1_right/walker_s2_description_hand3_v1_left_hand3_v1_right.urdf"
URDF_PATH="${WALKER_S2_URDF:-$DEFAULT_URDF}"
POLICY_PATH="${WALKER_GRASP_POLICY_PATH:-${SCRIPT_DIR}/recordings/train/walker_s2_grasp_act_smoke/checkpoints/001000/pretrained_model}"

if [[ ! -f "$URDF_PATH" ]]; then
  echo "Walker S2 URDF was not found: $URDF_PATH" >&2
  echo "Set WALKER_S2_URDF=/absolute/path/to/the/robot.urdf." >&2
  exit 1
fi

if [[ ! -d "$POLICY_PATH" ]]; then
  echo "Policy checkpoint was not found: $POLICY_PATH" >&2
  echo "Set WALKER_GRASP_POLICY_PATH=/absolute/path/to/pretrained_model." >&2
  exit 1
fi

exec "$ISAAC_PYTHON" scripts/eval_walker_grasp_policy.py \
  --urdf "$URDF_PATH" \
  --policy-path "$POLICY_PATH" \
  "$@"
