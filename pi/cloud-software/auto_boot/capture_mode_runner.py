#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
CLOUD_SOFTWARE_DIR = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "capture_mode_config.json"
AUTO_CAPTURE_PATH = CLOUD_SOFTWARE_DIR / "auto_capture_cloud.py"


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def log(message: str, log_path: Path | None) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"{timestamp} {message}"
    print(line, flush=True)
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def clear_wakealarm() -> None:
    run_cmd(["bash", "-lc", "echo 0 | sudo tee /sys/class/rtc/rtc0/wakealarm >/dev/null"])


def set_wakealarm(seconds_from_now: int) -> None:
    clear_wakealarm()
    run_cmd(
        [
            "bash",
            "-lc",
            f"echo +{int(seconds_from_now)} | sudo tee /sys/class/rtc/rtc0/wakealarm >/dev/null",
        ]
    )


def shutdown_now() -> None:
    run_cmd(["sync"])
    run_cmd(["sudo", "halt"])


def auto_capture_cmd(config: dict[str, Any], mode_config: dict[str, Any]) -> list[str]:
    python_bin = str(config.get("python_bin", sys.executable))
    return [python_bin, str(AUTO_CAPTURE_PATH), *map(str, mode_config["auto_capture_args"])]


def run_boot_mode(config: dict[str, Any], log_path: Path | None) -> None:
    boot_config = config["boot"]
    cmd = auto_capture_cmd(config, boot_config)
    log(f"Starting boot mode: {' '.join(cmd)}", log_path)
    run_cmd(cmd, cwd=CLOUD_SOFTWARE_DIR)


def run_low_power_mode(
    config: dict[str, Any],
    log_path: Path | None,
    no_shutdown: bool,
) -> None:
    low_power_config = config["low_power"]
    wake_seconds = int(low_power_config.get("wake_seconds", 3600))
    cmd = auto_capture_cmd(config, low_power_config)

    log(f"Starting low_power capture: {' '.join(cmd)}", log_path)
    try:
        run_cmd(cmd, cwd=CLOUD_SOFTWARE_DIR)
    finally:
        log(f"Scheduling RTC wake in {wake_seconds} seconds", log_path)
        set_wakealarm(wake_seconds)

    if no_shutdown or not bool(low_power_config.get("shutdown", True)):
        log("Skipping shutdown", log_path)
        return

    log("Halting Raspberry Pi for low-power interval", log_path)
    shutdown_now()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the configured cloud capture mode for systemd boot startup."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to capture_mode_config.json",
    )
    parser.add_argument(
        "--mode",
        choices=["configured", "boot", "low_power"],
        default="configured",
        help="Mode to run. Use 'configured' to read the mode from the config file.",
    )
    parser.add_argument(
        "--no-shutdown",
        action="store_true",
        help="In low_power mode, schedule RTC wake but do not halt. Useful for testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    log_path = (
        resolve_path(config["log_path"])
        if config.get("log_path")
        else None
    )
    mode = config.get("mode", "manual") if args.mode == "configured" else args.mode

    log(f"Loaded config {config_path}; selected mode={mode}", log_path)
    if mode == "manual":
        log("Manual mode selected; nothing to run at boot", log_path)
        return
    if mode == "boot":
        run_boot_mode(config, log_path)
        return
    if mode == "low_power":
        run_low_power_mode(config, log_path, no_shutdown=args.no_shutdown)
        return

    raise ValueError(f"Unknown capture mode: {mode}")


if __name__ == "__main__":
    main()
