#!/usr/bin/env bash
# Launch Part Sorting keyboard teleop using Isaac Sim's Python (container or native).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -x /isaac-sim/python.sh ]]; then
  ISAAC_PYTHON=(/isaac-sim/python.sh)
elif [[ -x /home/chris/isaacsim/python.sh ]]; then
  ISAAC_PYTHON=(/home/chris/isaacsim/python.sh)
else
  echo "Isaac Sim python.sh not found (tried /isaac-sim and ~/isaacsim)." >&2
  exit 1
fi

RESOURCES="${SCRIPT_DIR}/assets/resources"
mkdir -p "$RESOURCES"
if [[ ! -e "${RESOURCES}/WalkerS2-Model" && -d "${SCRIPT_DIR}/../WalkerS2-Model" ]]; then
  ln -sf "../../../WalkerS2-Model" "${RESOURCES}/WalkerS2-Model"
fi

# Writable pip cache (avoids /root/.cache permission warnings in Docker).
export PIP_CACHE_DIR="${SCRIPT_DIR}/.cache/pip"
mkdir -p "$PIP_CACHE_DIR"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

if [[ -d /workspace/WalkerS2-Model ]]; then
  export ZOLLENT_REPO_ROOT=/workspace
elif [[ -d "${SCRIPT_DIR}/../WalkerS2-Model" ]]; then
  export ZOLLENT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

URDF_REL="WalkerS2-Model/walker_s2_official/walker_s2.urdf"
URDF_CANDIDATES=(
  "${ZOLLENT_REPO_ROOT:-}/${URDF_REL}"
  "${SCRIPT_DIR}/assets/resources/${URDF_REL}"
)
URDF_FOUND=""
for candidate in "${URDF_CANDIDATES[@]}"; do
  if [[ -f "$candidate" ]]; then
    URDF_FOUND="$candidate"
    break
  fi
done
if [[ -z "$URDF_FOUND" ]]; then
  echo "Missing URDF: ${URDF_REL}" >&2
  echo "Run on the host: python3 scripts/setup_official_walker_s2.py" >&2
  echo "If using Docker, restart the container via ./run.sh so WalkerS2-Model is mounted." >&2
  exit 1
fi

# Refresh editable install only when project metadata changed (skip slow pip on every run).
EDITABLE_STAMP="${SCRIPT_DIR}/.cache/editable-install.stamp"
mkdir -p "${SCRIPT_DIR}/.cache"
if [[ ! -f "$EDITABLE_STAMP" ]] \
   || [[ pyproject.toml -nt "$EDITABLE_STAMP" ]] \
   || [[ setup.py -nt "$EDITABLE_STAMP" ]]; then
  "${ISAAC_PYTHON[@]}" -m pip install -e . --no-deps -q
  touch "$EDITABLE_STAMP"
fi

exec "${ISAAC_PYTHON[@]}" -m lerobot.scripts.lerobot_teleoperate \
  --robot.type=walker_s2_sim \
  --robot.headless=false \
  --robot.enable_sim_cameras=false \
  --teleop.type=walker_s2_keyboard \
  --task=Part_Sorting \
  --display_data=false \
  "$@"
