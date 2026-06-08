#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/capture_mode_config.json}"
SERVICE_NAME="${SERVICE_NAME:-cloud-capture-mode.service}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
TARGET_SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

usage() {
  cat <<USAGE
Usage:
  ./configure_capture_mode.sh manual
  ./configure_capture_mode.sh boot
  ./configure_capture_mode.sh low_power
  ./configure_capture_mode.sh status

Modes:
  manual     Disable boot automation. Run auto_capture_cloud.py manually.
  boot       Run auto_capture_cloud.py automatically after boot.
  low_power  Run one capture after boot, schedule RTC wake, then halt.

Environment overrides:
  CONFIG_PATH=/path/to/capture_mode_config.json
  PYTHON_BIN=/usr/bin/python3
  SERVICE_NAME=cloud-capture-mode.service
USAGE
}

set_mode() {
  local mode="$1"
  "${PYTHON_BIN}" - "$CONFIG_PATH" "$mode" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
mode = sys.argv[2]
data = json.loads(path.read_text(encoding="utf-8"))
data["mode"] = mode
path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
PY
}

install_service() {
  sudo tee "${TARGET_SERVICE_PATH}" >/dev/null <<EOF
[Unit]
Description=Cloud capture mode runner
After=local-fs.target systemd-modules-load.service network-online.target
Wants=local-fs.target

[Service]
Type=oneshot
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${PYTHON_BIN} "${SCRIPT_DIR}/capture_mode_runner.py" --config "${CONFIG_PATH}"
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

  sudo systemctl daemon-reload
}

show_status() {
  echo "Config: ${CONFIG_PATH}"
  "${PYTHON_BIN}" - "$CONFIG_PATH" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"Configured mode: {data.get('mode', 'manual')}")
print(f"Boot args: {' '.join(data.get('boot', {}).get('auto_capture_args', []))}")
low = data.get("low_power", {})
print(f"Low-power wake_seconds: {low.get('wake_seconds')}")
print(f"Low-power args: {' '.join(low.get('auto_capture_args', []))}")
PY
  systemctl is-enabled "${SERVICE_NAME}" 2>/dev/null || true
  systemctl status "${SERVICE_NAME}" --no-pager -l 2>/dev/null || true
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

case "$1" in
  manual)
    set_mode manual
    install_service
    sudo systemctl disable "${SERVICE_NAME}" >/dev/null || true
    echo "Mode set to manual. Boot automation is disabled."
    ;;
  boot)
    set_mode boot
    install_service
    sudo systemctl enable "${SERVICE_NAME}" >/dev/null
    echo "Mode set to boot. Service enabled: ${SERVICE_NAME}"
    ;;
  low_power)
    set_mode low_power
    install_service
    sudo systemctl enable "${SERVICE_NAME}" >/dev/null
    echo "Mode set to low_power. Service enabled: ${SERVICE_NAME}"
    ;;
  status)
    show_status
    ;;
  *)
    usage
    exit 1
    ;;
esac

echo
echo "Test without reboot:"
echo "  sudo systemctl start ${SERVICE_NAME}"
echo
echo "Logs:"
echo "  journalctl -u ${SERVICE_NAME} -n 100 --no-pager"
