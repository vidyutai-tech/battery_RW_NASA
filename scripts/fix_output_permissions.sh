#!/usr/bin/env bash
# Fix write permissions for training/optimization outputs.
#
# Run once (requires sudo — skip if you are not in sudoers):
#   sudo bash scripts/fix_output_permissions.sh hima
#
# Without sudo, use a user-owned output directory instead:
#   python scripts/03_optimize_profile_families.py --out_dir outputs/charging_opt_user/hima

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER="${1:-${SUDO_USER:-$(whoami)}}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Re-run with sudo: sudo bash scripts/fix_output_permissions.sh [username]"
  exit 1
fi

echo "Setting ownership of outputs/ and .mplconfig/ to ${USER}:${USER}"
mkdir -p "${ROOT}/outputs" "${ROOT}/.mplconfig"
chown -R "${USER}:${USER}" "${ROOT}/outputs" "${ROOT}/.mplconfig"
chmod -R u+rwX "${ROOT}/outputs" "${ROOT}/.mplconfig"

if command -v setfacl >/dev/null 2>&1; then
  echo "Applying default ACLs so ${USER} keeps write access on new files"
  setfacl -R -m "u:${USER}:rwx" "${ROOT}/outputs"
  setfacl -R -d -m "u:${USER}:rwx" "${ROOT}/outputs"
fi

echo "Done. Verify:"
ls -la "${ROOT}/outputs/charging_opt/models/stage3_optimization/" | head
