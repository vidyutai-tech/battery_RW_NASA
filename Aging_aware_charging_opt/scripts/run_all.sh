#!/usr/bin/env bash
# Convenience driver. Each stage is independently re-runnable.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/Aging_aware_charging_opt/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${ROOT}/venv/bin/python"
DEVICE="${1:-auto}"
if [[ "${1:-}" == --device ]]; then
  DEVICE="${2:-auto}"
fi

run() {
  echo
  echo ">>>>>>>> $*"
  "$PY" "$@"
}

run Aging_aware_charging_opt/scripts/00_verify_environment.py
run Aging_aware_charging_opt/scripts/02_verify_bdt.py --device "$DEVICE"
run Aging_aware_charging_opt/scripts/05_build_reward_anchors.py --device "$DEVICE"
run Aging_aware_charging_opt/scripts/06_sanity_checks.py --device "$DEVICE"
run Aging_aware_charging_opt/scripts/07_run_random_search.py --device "$DEVICE"
run Aging_aware_charging_opt/scripts/08_run_gp_bo.py --device "$DEVICE"
echo
echo "Optimization stages finished."
