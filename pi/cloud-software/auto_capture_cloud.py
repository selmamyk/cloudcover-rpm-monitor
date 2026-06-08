#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import gpiod
from gpiod.line import Direction, Value
from smbus2 import SMBus

from image_processing import (
    measure_frame_quality,
    normalize_hyta_method,
    predict_hyta_for_bgr,
    raw10_to_preview_bgr,
    save_bgr_image,
    save_capture_debug_outputs,
    save_debug_preview,
    unpack_raw10,
)
from epaper_status import DEFAULT_EPAPER_LIB_PATH, update_cloud_status_display


I2C_BUS = 1
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMXCONFIG_PATH = str((SCRIPT_DIR.parent / "driver" / "imxconfig.sh").resolve())
LIGHT_SENSOR_ADDR = 0x38
REG_MODE_CONTROL = 0x41
REG_ALS_PS_CONTROL = 0x42
REG_ALS_DATA0_LSB = 0x46
REG_ALS_DATA0_MSB = 0x47
REG_ALS_DATA1_LSB = 0x48
REG_ALS_DATA1_MSB = 0x49
REG_MANUFACT_ID = 0x92
EXPECTED_MANUFACT_ID = 0xE0

IMX_PINS = {
    "SEQ": 26,
    "LS": 6,
    "CLK": 5,
    "XCLR": 25,
}

FILTER_PINS = {
    "HB+": 17,
    "HB-": 27,
}

DEFAULT_CTRL_FALLBACKS = {
    "exposure": {"min": 1, "max": 1123, "step": 1, "default": 1123, "value": 30},
    "vertical_blanking": {"min": 45, "max": 129991, "step": 1, "default": 45, "value": 45},
    "analogue_gain": {"min": 0, "max": 300, "step": 1, "default": 0, "value": 0},
    "gain": {"min": 0, "max": 420, "step": 1, "default": 0, "value": 0},
}

# Conservative policies for cloud-cover capture. We prefer preserving
# highlight structure and low noise over making the preview look bright.
EXPOSURE_POLICIES = {
    "day": {
        "description": "low-noise daylight capture with highlight protection",
        "max_exposure": 80,
        "max_analogue_gain": 10,
        "max_gain": 0,
        "target_mean_min": 80,
        "target_mean_max": 165,
        "target_p99_min": 180,
        "target_p99_max": 500,
        "max_clipped_fraction": 0.003,
        "max_dark_fraction": 0.65,
    },
    "twilight": {
        "description": "longer exposure before gain for dusk and dawn",
        "max_exposure": 360,
        "max_analogue_gain": 100,
        "max_gain": 0,
        "target_mean_min": 75,
        "target_mean_max": 170,
        "target_p99_min": 160,
        "target_p99_max": 520,
        "max_clipped_fraction": 0.004,
        "max_dark_fraction": 0.75,
    },
    "night": {
        "description": "night capture; digital gain is allowed only after exposure and analogue gain",
        "max_exposure": 10000,
        "max_analogue_gain": 200,
        "max_gain": 0,
        "target_mean_min": 35,
        "target_mean_max": 120,
        "target_p99_min": 80,
        "target_p99_max": 600,
        "max_clipped_fraction": 0.01,
        "max_dark_fraction": 0.95,
    },
}

DEBUG_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="2">
  <title>Cloud Auto Capture Debug</title>
  <style>
    body { background: #101418; color: #f3f4f6; font-family: Arial, sans-serif; margin: 0; }
    .wrap { max-width: 1200px; margin: 0 auto; padding: 20px; }
    img { width: 100%; height: auto; border: 1px solid #334155; background: #000; }
    .meta { color: #cbd5e1; margin-bottom: 16px; }
    code { background: #1e293b; padding: 2px 6px; border-radius: 4px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Cloud Auto Capture Debug</h1>
    <p class="meta">This page refreshes every 2 seconds. Latest captured HYTA debug:</p>
    <img src="latest_debug.jpg" alt="Latest captured HYTA debug">
    <p class="meta" style="margin-top:24px;">Latest captured image:</p>
    <img src="latest_capture.jpg" alt="Latest captured preview">
  </div>
</body>
</html>
"""


@dataclass
class ControlRange:
    name: str
    minimum: int
    maximum: int
    step: int
    default: int
    value: int

    def clamp(self, raw_value: float | int) -> int:
        value = int(round(raw_value))
        value = max(self.minimum, min(self.maximum, value))
        if self.step > 1:
            value = self.minimum + ((value - self.minimum) // self.step) * self.step
        return value


def run_cmd(cmd: list[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=capture_output)


def detect_host_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "PI-IP"


def read_u16(bus: SMBus, addr: int, reg_lsb: int, reg_msb: int) -> int:
    lsb = bus.read_byte_data(addr, reg_lsb)
    msb = bus.read_byte_data(addr, reg_msb)
    return (msb << 8) | lsb


def read_light_sensor() -> dict:
    with SMBus(I2C_BUS) as bus:
        mid = bus.read_byte_data(LIGHT_SENSOR_ADDR, REG_MANUFACT_ID)
        if mid != EXPECTED_MANUFACT_ID:
            raise RuntimeError(
                f"Unexpected light sensor ID 0x{mid:02x}, expected 0x{EXPECTED_MANUFACT_ID:02x}"
            )

        bus.write_byte_data(LIGHT_SENSOR_ADDR, REG_MODE_CONTROL, 0x89)
        bus.write_byte_data(LIGHT_SENSOR_ADDR, REG_ALS_PS_CONTROL, 0x02)
        time.sleep(0.2)

        data0 = read_u16(bus, LIGHT_SENSOR_ADDR, REG_ALS_DATA0_LSB, REG_ALS_DATA0_MSB)
        data1 = read_u16(bus, LIGHT_SENSOR_ADDR, REG_ALS_DATA1_LSB, REG_ALS_DATA1_MSB)

    # Simple proxy for scene brightness. This is sufficient for profile selection.
    brightness = max(data0 - data1, 0)
    return {
        "manufacturer_id": mid,
        "als_data0": data0,
        "als_data1": data1,
        "brightness_proxy": brightness,
    }


def pulse_gpio(pins: dict[str, int], active_pin: str, inactive_pin: str | None = None) -> None:
    config = {
        pin: gpiod.LineSettings(
            direction=Direction.OUTPUT,
            output_value=Value.INACTIVE,
            active_low=False,
        )
        for pin in pins.values()
    }

    with gpiod.Chip("/dev/gpiochip0") as chip:
        with chip.request_lines(config, consumer="cloud-capture") as req:
            req.set_value(pins[active_pin], Value.ACTIVE)
            if inactive_pin is not None:
                req.set_value(pins[inactive_pin], Value.INACTIVE)
            time.sleep(0.15)
            req.set_value(pins[active_pin], Value.INACTIVE)
            if inactive_pin is not None:
                req.set_value(pins[inactive_pin], Value.INACTIVE)
            time.sleep(0.1)


def power_on_sensor() -> None:
    config = {
        pin: gpiod.LineSettings(
            direction=Direction.OUTPUT,
            output_value=Value.INACTIVE,
            active_low=False,
        )
        for pin in IMX_PINS.values()
    }

    with gpiod.Chip("/dev/gpiochip0") as chip:
        with chip.request_lines(config, consumer="imx-power-on") as req:
            req.set_value(IMX_PINS["SEQ"], Value.ACTIVE)
            time.sleep(0.2)
            req.set_value(IMX_PINS["LS"], Value.ACTIVE)
            time.sleep(0.1)
            req.set_value(IMX_PINS["CLK"], Value.ACTIVE)
            time.sleep(0.2)
            req.set_value(IMX_PINS["XCLR"], Value.ACTIVE)
            time.sleep(0.1)


def power_off_sensor() -> None:
    config = {
        pin: gpiod.LineSettings(
            direction=Direction.OUTPUT,
            output_value=Value.INACTIVE,
            active_low=False,
        )
        for pin in IMX_PINS.values()
    }

    with gpiod.Chip("/dev/gpiochip0") as chip:
        with chip.request_lines(config, consumer="imx-power-off") as req:
            req.set_value(IMX_PINS["XCLR"], Value.INACTIVE)
            time.sleep(0.1)
            req.set_value(IMX_PINS["CLK"], Value.INACTIVE)
            time.sleep(0.1)
            req.set_value(IMX_PINS["LS"], Value.INACTIVE)
            time.sleep(0.1)
            req.set_value(IMX_PINS["SEQ"], Value.INACTIVE)
            time.sleep(0.1)


def set_ir_cut(enabled: bool) -> str:
    if enabled:
        pulse_gpio(FILTER_PINS, "HB+", "HB-")
        return "on"
    pulse_gpio(FILTER_PINS, "HB-", "HB+")
    return "off"


def parse_control_ranges(text: str, names: list[str]) -> dict[str, ControlRange]:
    ranges: dict[str, ControlRange] = {}
    for name in names:
        pattern = (
            rf"^\s*{re.escape(name)}\s+.*?min=(-?\d+)\s+max=(-?\d+)\s+step=(-?\d+)\s+"
            rf"default=(-?\d+)\s+value=(-?\d+)"
        )
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            minimum, maximum, step, default, value = map(int, match.groups())
            ranges[name] = ControlRange(name, minimum, maximum, step, default, value)
            continue

        fallback = DEFAULT_CTRL_FALLBACKS[name]
        ranges[name] = ControlRange(
            name=name,
            minimum=fallback["min"],
            maximum=fallback["max"],
            step=fallback["step"],
            default=fallback["default"],
            value=fallback["value"],
        )
    return ranges


def get_control_ranges(subdev: str) -> dict[str, ControlRange]:
    result = run_cmd(["v4l2-ctl", "-d", subdev, "--list-ctrls"], capture_output=True)
    return parse_control_ranges(
        result.stdout,
        ["exposure", "vertical_blanking", "analogue_gain", "gain"],
    )


def wait_for_control_ranges(
    subdev: str,
    timeout_sec: float,
    retry_interval_sec: float,
) -> dict[str, ControlRange]:
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            return get_control_ranges(subdev)
        except subprocess.CalledProcessError as exc:
            if time.monotonic() >= deadline:
                stderr = (exc.stderr or "").strip()
                stdout = (exc.stdout or "").strip()
                details = stderr or stdout or str(exc)
                raise RuntimeError(
                    f"Could not read controls from {subdev} after {timeout_sec:.1f}s: {details}"
                ) from exc
            time.sleep(retry_interval_sec)


def get_control_values(subdev: str) -> dict[str, int]:
    result = run_cmd(
        [
            "v4l2-ctl",
            "-d",
            subdev,
            "--get-ctrl=exposure,vertical_blanking,gain,analogue_gain",
        ],
        capture_output=True,
    )
    values: dict[str, int] = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        try:
            values[key.strip()] = int(raw_value.strip())
        except ValueError:
            continue
    return values


def set_controls(subdev: str, exposure: int, analogue_gain: int, gain: int) -> None:
    ctrl_arg = f"exposure={exposure},analogue_gain={analogue_gain},gain={gain}"
    run_cmd(["v4l2-ctl", "-d", subdev, f"--set-ctrl={ctrl_arg}"])


def set_vertical_blanking(subdev: str, vertical_blanking: int) -> None:
    run_cmd(["v4l2-ctl", "-d", subdev, f"--set-ctrl=vertical_blanking={vertical_blanking}"])


def required_vertical_blanking(
    exposure: int,
    height: int,
    vertical_blanking_range: ControlRange,
) -> int:
    # IMX385 exposure max is approximately frame_length - 2.
    required = int(exposure) - int(height) + 2
    return vertical_blanking_range.clamp(max(vertical_blanking_range.minimum, required))


def capture_frame(
    video_device: str,
    output_path: Path,
    width: int = 1920,
    height: int = 1080,
    pixelformat: str = "pRAA",
    mmap_buffers: int = 4,
    skip: int = 2,
) -> None:
    run_cmd(
        [
            "v4l2-ctl",
            "-d",
            video_device,
            f"--set-fmt-video=width={width},height={height},pixelformat={pixelformat}",
            f"--stream-mmap={mmap_buffers}",
            f"--stream-skip={skip}",
            "--stream-count=1",
            f"--stream-to={output_path}",
        ]
    )


def start_debug_server(base_dir: Path, port: int) -> tuple[ThreadingHTTPServer, threading.Thread]:
    class QuietHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(base_dir), **kwargs)

        def log_message(self, format: str, *args) -> None:
            return

    server = ThreadingHTTPServer(("0.0.0.0", port), QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def choose_start_profile(
    light_stats: dict,
    ranges: dict[str, ControlRange],
    force_ir: str,
) -> dict:
    brightness = light_stats["brightness_proxy"]

    if brightness >= 12000:
        profile = {
            "mode": "day",
            "exposure": 3,
            "analogue_gain": 0,
            "gain": 0,
            "ir_cut": "on",
        }
    elif brightness >= 5000:
        profile = {
            "mode": "day",
            "exposure": 6,
            "analogue_gain": 0,
            "gain": 0,
            "ir_cut": "on",
        }
    elif brightness >= 1500:
        profile = {
            "mode": "day",
            "exposure": 12,
            "analogue_gain": 0,
            "gain": 0,
            "ir_cut": "on",
        }
    elif brightness >= 400:
        profile = {
            "mode": "day",
            "exposure": 40,
            "analogue_gain": 0,
            "gain": 0,
            "ir_cut": "on",
        }
    elif brightness >= 120:
        profile = {
            "mode": "twilight",
            "exposure": 120,
            "analogue_gain": 5,
            "gain": 0,
            "ir_cut": "off",
        }
    elif brightness >= 30:
        profile = {
            "mode": "twilight",
            "exposure": 280,
            "analogue_gain": 15,
            "gain": 0,
            "ir_cut": "off",
        }
    else:
        profile = {
            "mode": "night",
            "exposure": 600,
            "analogue_gain": 30,
            "gain": 0,
            "ir_cut": "off",
        }

    if force_ir in {"on", "off"}:
        profile["ir_cut"] = force_ir

    policy = EXPOSURE_POLICIES[profile["mode"]]
    return {
        "mode": profile["mode"],
        "exposure": ranges["exposure"].clamp(profile["exposure"]),
        "analogue_gain": ranges["analogue_gain"].clamp(profile["analogue_gain"]),
        "gain": ranges["gain"].clamp(profile["gain"]),
        "ir_cut": profile["ir_cut"],
        "policy": policy,
    }


def clamp_to_policy(
    controls: dict,
    ranges: dict[str, ControlRange],
    policy: dict,
) -> dict:
    return {
        "exposure": ranges["exposure"].clamp(
            min(controls["exposure"], policy["max_exposure"])
        ),
        "analogue_gain": ranges["analogue_gain"].clamp(
            min(controls["analogue_gain"], policy["max_analogue_gain"])
        ),
        "gain": ranges["gain"].clamp(min(controls["gain"], policy["max_gain"])),
    }


def tune_exposure(
    video_device: str,
    subdev: str,
    width: int,
    height: int,
    ranges: dict[str, ControlRange],
    start_controls: dict,
    exposure_policy: dict,
    preview_path: Path,
    debug_dir: Path | None = None,
    debug_prefix: str = "tune",
    bayer_pattern: str = "BGGR",
    ir_cut_status: str | None = None,
    latest_debug_path: Path | None = None,
    enable_hyta_debug: bool = False,
    hyta_tf: float = 0.15,
    hyta_sigma_thr: float = 0.01,
    threshold_radius_scale: float = 0.50,
    prediction_radius_scale: float = 0.65,
    hyta_method: str = "hyta",
    bottom_boost: float = 0.125,
    bottom_boost_start: float = 0.55,
    bottom_boost_gate: float = 0.35,
    low_okta_base_cf_max: float = 0.25,
    low_okta_cap_okta: int = 1,
    max_iters: int = 5,
    auto_vertical_blanking: bool = True,
) -> tuple[dict, dict, list[dict]]:
    controls = clamp_to_policy(start_controls, ranges, exposure_policy)
    history: list[dict] = []
    exposure_max = int(exposure_policy["max_exposure"])
    analogue_gain_max = ranges["analogue_gain"].clamp(
        exposure_policy["max_analogue_gain"]
    )
    gain_max = ranges["gain"].clamp(exposure_policy["max_gain"])
    current_vertical_blanking: int | None = None

    for iter_idx in range(max_iters):
        if auto_vertical_blanking:
            wanted_vertical_blanking = required_vertical_blanking(
                exposure=controls["exposure"],
                height=height,
                vertical_blanking_range=ranges["vertical_blanking"],
            )
            if wanted_vertical_blanking != current_vertical_blanking:
                set_vertical_blanking(subdev, wanted_vertical_blanking)
                time.sleep(0.05)
                ranges = get_control_ranges(subdev)
                current_vertical_blanking = wanted_vertical_blanking

        controls["exposure"] = ranges["exposure"].clamp(controls["exposure"])
        set_controls(subdev, **controls)
        time.sleep(0.15)
        capture_frame(video_device, preview_path, width=width, height=height)
        raw10 = unpack_raw10(preview_path, width, height)
        quality = measure_frame_quality(raw10)
        history.append(
            {
                "controls": {
                    **controls,
                    "vertical_blanking": current_vertical_blanking,
                },
                "quality": quality,
            }
        )

        if debug_dir is not None:
            save_debug_preview(
                raw10=raw10,
                output_path=debug_dir / f"{debug_prefix}_iter_{iter_idx:02d}.png",
                controls=controls,
                quality=quality,
                bayer_pattern=bayer_pattern,
                ir_cut_status=ir_cut_status,
                latest_path=latest_debug_path,
                enable_hyta=enable_hyta_debug,
                hyta_tf=hyta_tf,
                hyta_sigma_thr=hyta_sigma_thr,
                threshold_radius_scale=threshold_radius_scale,
                prediction_radius_scale=prediction_radius_scale,
                hyta_method=hyta_method,
                bottom_boost=bottom_boost,
                bottom_boost_start=bottom_boost_start,
                bottom_boost_gate=bottom_boost_gate,
                low_okta_base_cf_max=low_okta_base_cf_max,
                low_okta_cap_okta=low_okta_cap_okta,
            )

        mean_value = quality["mean"]
        p99 = quality["p99"]
        dark_fraction = quality["dark_fraction"]
        clipped_fraction = quality["clipped_fraction"]

        target_mean_min = exposure_policy["target_mean_min"]
        target_mean_max = exposure_policy["target_mean_max"]
        target_p99_min = exposure_policy["target_p99_min"]
        target_p99_max = exposure_policy["target_p99_max"]
        max_dark_fraction = exposure_policy["max_dark_fraction"]
        max_clipped_fraction = exposure_policy["max_clipped_fraction"]

        if (
            target_mean_min <= mean_value <= target_mean_max
            and target_p99_min <= p99 <= target_p99_max
            and dark_fraction <= max_dark_fraction
            and clipped_fraction <= max_clipped_fraction
        ):
            break

        if (
            p99 > target_p99_max
            or mean_value > target_mean_max
            or clipped_fraction > max_clipped_fraction
        ):
            if controls["gain"] > ranges["gain"].minimum:
                controls["gain"] = ranges["gain"].clamp(
                    max(ranges["gain"].minimum, controls["gain"] - 20)
                )
            elif controls["analogue_gain"] > ranges["analogue_gain"].minimum:
                controls["analogue_gain"] = ranges["analogue_gain"].clamp(
                    controls["analogue_gain"] - 10
                )
            else:
                controls["exposure"] = ranges["exposure"].clamp(
                    max(ranges["exposure"].minimum, controls["exposure"] * 0.75)
                )
            continue

        if mean_value < target_mean_min or p99 < target_p99_min:
            if controls["exposure"] < exposure_max:
                if mean_value < 1:
                    factor = 4.0
                else:
                    factor = max(1.2, min(4.0, target_mean_min / mean_value))
                controls["exposure"] = int(round(min(exposure_max, controls["exposure"] * factor)))
            elif controls["analogue_gain"] < analogue_gain_max:
                controls["analogue_gain"] = ranges["analogue_gain"].clamp(
                    min(
                        analogue_gain_max,
                        controls["analogue_gain"] + (30 if mean_value < 40 else 10),
                    )
                )
            elif controls["gain"] < gain_max:
                controls["gain"] = ranges["gain"].clamp(
                    min(gain_max, controls["gain"] + (60 if mean_value < 40 else 20))
                )
            else:
                break
            continue

        break

    set_controls(subdev, **controls)
    return controls, history[-1]["quality"], history


def save_metadata(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_hyta_prediction_csv(path: Path, row: dict) -> None:
    fieldnames = [
        "image_id",
        "timestamp",
        "raw_file",
        "processed_image",
        "hyta_debug_image",
        "cloud_percent",
        "cloud_fraction",
        "okta",
        "okta_text",
        "threshold",
        "sigma",
        "hyta_method",
        "bottom_boost",
        "bottom_boost_start",
        "bottom_boost_gate",
        "low_okta_base_cf_max",
        "low_okta_cap_okta",
        "exposure",
        "analogue_gain",
        "gain",
        "ir_cut",
        "exposure_mode",
        "als_brightness_proxy",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def slugify_label(value: str) -> str:
    cleaned = re.sub(r"\s+", "_", value.strip())
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "", cleaned)
    return cleaned.strip("_-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture cloud-monitoring RAW images with automatic exposure, gain, and IR-cut control."
    )
    parser.add_argument("--output-dir", default="dataset", help="Base output directory")
    parser.add_argument(
        "--label",
        default="",
        help="Optional text appended to the run folder name, for example 'overskyet'",
    )
    parser.add_argument("--interval", type=float, default=300.0, help="Seconds between captures")
    parser.add_argument(
        "--num-images",
        type=int,
        default=0,
        help="Number of images to capture. Use 0 to keep capturing until stopped.",
    )
    parser.add_argument("--video-device", default="/dev/video0", help="Video node for frame capture")
    parser.add_argument(
        "--subdev-device",
        default="/dev/v4l-subdev2",
        help="Subdevice node used for exposure/gain controls",
    )
    parser.add_argument("--width", type=int, default=1920, help="RAW image width")
    parser.add_argument("--height", type=int, default=1080, help="RAW image height")
    parser.add_argument(
        "--retune-every",
        type=int,
        default=1,
        help="Re-run auto tuning every N captured images",
    )
    parser.add_argument(
        "--max-tune-iters",
        type=int,
        default=8,
        help="Maximum exposure/gain adjustment iterations when auto tuning",
    )
    parser.add_argument(
        "--auto-vblank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically raise vertical_blanking when auto exposure needs longer exposure",
    )
    parser.add_argument(
        "--ir-cut",
        choices=["auto", "on", "off"],
        default="auto",
        help="IR-cut filter handling",
    )
    parser.add_argument(
        "--power-on",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the IMX385 power-on sequence before capturing",
    )
    parser.add_argument(
        "--imxconfig-path",
        default=DEFAULT_IMXCONFIG_PATH,
        help="Optional path to imxconfig.sh to run after power-on and before V4L2 controls are read",
    )
    parser.add_argument(
        "--post-config-settle-seconds",
        type=float,
        default=0.3,
        help="Seconds to wait after --imxconfig-path before reading V4L2 controls",
    )
    parser.add_argument(
        "--device-ready-timeout",
        type=float,
        default=20.0,
        help="Seconds to retry reading V4L2 controls after boot/power-on",
    )
    parser.add_argument(
        "--device-ready-retry-interval",
        type=float,
        default=1.0,
        help="Seconds between V4L2 control-read retries",
    )
    parser.add_argument(
        "--power-off-at-end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the IMX385 power-off sequence after capturing",
    )
    parser.add_argument(
        "--manual-exposure",
        type=int,
        default=None,
        help="Override auto exposure with a fixed exposure value",
    )
    parser.add_argument(
        "--manual-analogue-gain",
        type=int,
        default=None,
        help="Override auto analogue gain with a fixed value",
    )
    parser.add_argument(
        "--manual-gain",
        type=int,
        default=None,
        help="Override auto digital gain with a fixed value",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.4,
        help="Extra wait time after IR-cut/profile changes",
    )
    parser.add_argument(
        "--bayer-pattern",
        choices=["BGGR", "RGGB", "GRBG", "GBRG"],
        default="BGGR",
        help="Bayer pattern used when generating debug preview PNGs",
    )
    parser.add_argument(
        "--debug-preview",
        action="store_true",
        help="Save preview PNGs for each auto-tuning step",
    )
    parser.add_argument(
        "--debug-web-port",
        type=int,
        default=8091,
        help="Port for the live debug web page when using --debug-preview",
    )
    parser.add_argument(
        "--debug-hyta",
        action="store_true",
        help="Run HYTA on each debug preview and show the result live in the debug page",
    )
    parser.add_argument(
        "--save-hyta-debug-image",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save a captured_hyta_debug.png overlay image when HYTA is run",
    )
    parser.add_argument(
        "--save-image-preview",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save a processed preview of the captured image when HYTA is run",
    )
    parser.add_argument(
        "--hyta-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run HYTA for each captured RAW file and write prediction rows to CSV without requiring debug PNGs",
    )
    parser.add_argument(
        "--hyta-tf",
        type=float,
        default=0.05,
        help="HYTA fixed threshold for unimodal debug classification",
    )
    parser.add_argument(
        "--hyta-sigma-thr",
        type=float,
        default=0.01,
        help="HYTA sigma threshold for debug classification",
    )
    parser.add_argument(
        "--threshold-radius-scale",
        type=float,
        default=0.55,
        help="Radius scale for the smaller HYTA circle used to calculate the threshold",
    )
    parser.add_argument(
        "--prediction-radius-scale",
        type=float,
        default=0.65,
        help="Radius scale for the larger HYTA circle used for cloud predictions",
    )
    parser.add_argument(
        "--hyta-method",
        choices=["hyta", "imx385_adapted"],
        default="imx385_adapted",
        help="HYTA variant to use for debug previews, CSV predictions, and e-paper output",
    )
    parser.add_argument(
        "--bottom-boost",
        type=float,
        default=0.125,
        help="IMX385-adapted: maximum threshold increase at the bottom of the image",
    )
    parser.add_argument(
        "--bottom-boost-start",
        type=float,
        default=0.55,
        help="IMX385-adapted: normalized y position where bottom boost starts",
    )
    parser.add_argument(
        "--bottom-boost-gate",
        type=float,
        default=0.35,
        help="IMX385-adapted: apply bottom boost only when base cloud fraction is at least this value",
    )
    parser.add_argument(
        "--low-okta-base-cf-max",
        type=float,
        default=0.25,
        help="IMX385-adapted: cap output only when base cloud fraction is at or below this value",
    )
    parser.add_argument(
        "--low-okta-cap-okta",
        type=int,
        default=1,
        help="IMX385-adapted: maximum okta allowed by the low-okta cap",
    )
    parser.add_argument(
        "--epaper-display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update the e-paper display after each HYTA prediction",
    )
    parser.add_argument(
        "--epaper-lib-path",
        default=DEFAULT_EPAPER_LIB_PATH,
        help="Path to the Waveshare e-paper Python library",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.hyta_method = normalize_hyta_method(args.hyta_method)

    if args.retune_every < 1:
        raise ValueError("--retune-every must be at least 1")
    if args.max_tune_iters < 1:
        raise ValueError("--max-tune-iters must be at least 1")

    if args.power_on:
        print("Powering on IMX385...")
        power_on_sensor()
        time.sleep(0.5)

    if args.imxconfig_path:
        imxconfig_path = Path(args.imxconfig_path).expanduser().resolve()
        print(f"Running IMX385 config: {imxconfig_path}")
        run_cmd(["bash", str(imxconfig_path)])
        time.sleep(args.post_config_settle_seconds)

    ranges = wait_for_control_ranges(
        args.subdev_device,
        timeout_sec=args.device_ready_timeout,
        retry_interval_sec=args.device_ready_retry_interval,
    )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_stamp
    if args.label:
        label = slugify_label(args.label)
        if label:
            run_name = f"{run_stamp}_{label}"
    base_dir = Path(args.output_dir) / run_name
    base_dir.mkdir(parents=True, exist_ok=True)
    debug_server: ThreadingHTTPServer | None = None

    if args.debug_preview:
        (base_dir / "index.html").write_text(DEBUG_INDEX_HTML, encoding="utf-8")
        debug_server, _ = start_debug_server(base_dir, args.debug_web_port)
        print(f"Debug page: http://{detect_host_ip()}:{args.debug_web_port}/")

    preview_path = Path(f"/tmp/cloud_auto_preview_{os.getpid()}.raw")
    active_controls: dict | None = None
    active_filter: str | None = None
    active_profile_mode: str | None = None

    print(f"Saving dataset to: {base_dir}")
    print("Camera pipeline: raw10/v4l2")

    try:
        index = 0
        while args.num_images == 0 or index < args.num_images:
            sample_dir = base_dir / f"sample_{index:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            image_path = sample_dir / "image.raw"
            metadata_path = sample_dir / "metadata.json"

            retune_now = active_controls is None or (index % args.retune_every == 0)
            light_stats = read_light_sensor()
            profile = choose_start_profile(light_stats, ranges, args.ir_cut)

            if args.manual_exposure is not None:
                profile["exposure"] = ranges["exposure"].clamp(args.manual_exposure)
            if args.manual_analogue_gain is not None:
                profile["analogue_gain"] = ranges["analogue_gain"].clamp(args.manual_analogue_gain)
            if args.manual_gain is not None:
                profile["gain"] = ranges["gain"].clamp(args.manual_gain)

            if active_filter != profile["ir_cut"]:
                active_filter = set_ir_cut(profile["ir_cut"] == "on")
                time.sleep(args.settle_time)
                active_controls = None
                active_profile_mode = None

            if active_profile_mode is not None and active_profile_mode != profile["mode"]:
                active_controls = None
                active_profile_mode = None

            tuning_history: list[dict] = []
            preview_quality: dict | None = None
            capture_hyta_result: dict | None = None
            manual_controls_requested = (
                args.manual_exposure is not None
                or args.manual_analogue_gain is not None
                or args.manual_gain is not None
            )

            if (
                retune_now
                and args.manual_exposure is None
                and args.manual_analogue_gain is None
                and args.manual_gain is None
            ):
                debug_dir = sample_dir / "debug_tuning" if args.debug_preview else None
                start_controls = active_controls if active_controls is not None else profile
                active_controls, preview_quality, tuning_history = tune_exposure(
                    video_device=args.video_device,
                    subdev=args.subdev_device,
                    width=args.width,
                    height=args.height,
                    ranges=ranges,
                    start_controls=start_controls,
                    exposure_policy=profile["policy"],
                    preview_path=preview_path,
                    debug_dir=debug_dir,
                    debug_prefix=f"sample_{index:04d}",
                    bayer_pattern=args.bayer_pattern,
                    ir_cut_status=active_filter,
                    latest_debug_path=None,
                    enable_hyta_debug=args.debug_hyta,
                    hyta_tf=args.hyta_tf,
                    hyta_sigma_thr=args.hyta_sigma_thr,
                    threshold_radius_scale=args.threshold_radius_scale,
                    prediction_radius_scale=args.prediction_radius_scale,
                    hyta_method=args.hyta_method,
                    bottom_boost=args.bottom_boost,
                    bottom_boost_start=args.bottom_boost_start,
                    bottom_boost_gate=args.bottom_boost_gate,
                    low_okta_base_cf_max=args.low_okta_base_cf_max,
                    low_okta_cap_okta=args.low_okta_cap_okta,
                    max_iters=args.max_tune_iters,
                    auto_vertical_blanking=args.auto_vblank,
                )
                active_profile_mode = profile["mode"]
            else:
                if active_controls is None:
                    active_controls = {
                        "exposure": profile["exposure"],
                        "analogue_gain": profile["analogue_gain"],
                        "gain": profile["gain"],
                    }
                    active_profile_mode = profile["mode"]
                set_controls(args.subdev_device, **active_controls)
                time.sleep(0.15)

            print(
                f"[{index + 1}/{'unlimited' if args.num_images == 0 else args.num_images}] "
                f"ALS={light_stats['brightness_proxy']} "
                f"mode={profile['mode']} "
                f"IR={active_filter} "
                f"exp={active_controls['exposure']} "
                f"again={active_controls['analogue_gain']} "
                f"gain={active_controls['gain']}"
            )

            capture_frame(
                args.video_device,
                image_path,
                width=args.width,
                height=args.height,
            )
            final_controls = get_control_values(args.subdev_device)

            processed_image_path = None
            save_main_preview = args.debug_preview or args.save_hyta_debug_image
            if save_main_preview:
                final_controls_for_debug = {
                    "exposure": final_controls.get("exposure", active_controls["exposure"]),
                    "analogue_gain": final_controls.get("analogue_gain", active_controls["analogue_gain"]),
                    "gain": final_controls.get("gain", active_controls["gain"]),
                }
                final_quality = save_capture_debug_outputs(
                    raw_path=image_path,
                    sample_dir=sample_dir,
                    base_dir=base_dir,
                    width=args.width,
                    height=args.height,
                    bayer_pattern=args.bayer_pattern,
                    controls=final_controls_for_debug,
                    ir_cut_status=active_filter,
                    enable_hyta=args.debug_hyta or args.save_hyta_debug_image,
                    hyta_tf=args.hyta_tf,
                    hyta_sigma_thr=args.hyta_sigma_thr,
                    threshold_radius_scale=args.threshold_radius_scale,
                    prediction_radius_scale=args.prediction_radius_scale,
                    hyta_method=args.hyta_method,
                    bottom_boost=args.bottom_boost,
                    bottom_boost_start=args.bottom_boost_start,
                    bottom_boost_gate=args.bottom_boost_gate,
                    low_okta_base_cf_max=args.low_okta_base_cf_max,
                    low_okta_cap_okta=args.low_okta_cap_okta,
                    save_image_preview=args.debug_preview,
                    save_hyta_debug_image=args.save_hyta_debug_image or args.debug_hyta,
                )
                if preview_quality is None:
                    preview_quality = final_quality
                capture_hyta_result = final_quality.get("hyta")

            if args.hyta_csv and capture_hyta_result is None:
                raw10 = unpack_raw10(image_path, args.width, args.height)
                final_quality = measure_frame_quality(raw10)
                processed_bgr = raw10_to_preview_bgr(raw10, bayer_pattern=args.bayer_pattern)
                if args.save_image_preview:
                    processed_image_path = sample_dir / "processed_image.png"
                    save_bgr_image(processed_bgr, processed_image_path)
                    save_bgr_image(processed_bgr, base_dir / "latest_capture.jpg")
                capture_hyta_result = predict_hyta_for_bgr(
                    preview=processed_bgr,
                    hyta_tf=args.hyta_tf,
                    hyta_sigma_thr=args.hyta_sigma_thr,
                    threshold_radius_scale=args.threshold_radius_scale,
                    prediction_radius_scale=args.prediction_radius_scale,
                    hyta_method=args.hyta_method,
                    bottom_boost=args.bottom_boost,
                    bottom_boost_start=args.bottom_boost_start,
                    bottom_boost_gate=args.bottom_boost_gate,
                    low_okta_base_cf_max=args.low_okta_base_cf_max,
                    low_okta_cap_okta=args.low_okta_cap_okta,
                )
                final_quality["hyta"] = capture_hyta_result
                if preview_quality is None:
                    preview_quality = final_quality

            if args.save_image_preview and capture_hyta_result is not None and processed_image_path is None:
                raw10 = unpack_raw10(image_path, args.width, args.height)
                processed_bgr = raw10_to_preview_bgr(raw10, bayer_pattern=args.bayer_pattern)
                processed_image_path = sample_dir / "processed_image.png"
                save_bgr_image(processed_bgr, processed_image_path)
                save_bgr_image(processed_bgr, base_dir / "latest_capture.jpg")

            prediction_time = datetime.now()
            metadata = {
                "image_id": f"{index:04d}",
                "timestamp": prediction_time.isoformat(),
                "image_file": str(image_path),
                "raw_file": str(image_path),
                "camera_pipeline": "raw10/v4l2",
                "video_device": args.video_device,
                "subdev_device": args.subdev_device,
                "width": args.width,
                "height": args.height,
                "interval_sec": args.interval,
                "retune_every": args.retune_every,
                "auto_vblank": args.auto_vblank,
                "light_sensor": light_stats,
                "ir_cut": active_filter,
                "selected_profile": profile,
                "exposure_mode": profile["mode"],
                "exposure_policy": profile["policy"],
                "control_priority": "low-noise: exposure -> analogue_gain -> digital_gain",
                "selected_controls_before_capture": active_controls,
                "applied_controls": final_controls,
                "preview_quality": preview_quality,
                "hyta_prediction": capture_hyta_result,
                "hyta_settings": {
                    "method": args.hyta_method,
                    "hyta_tf": args.hyta_tf,
                    "hyta_sigma_thr": args.hyta_sigma_thr,
                    "threshold_radius_scale": args.threshold_radius_scale,
                    "prediction_radius_scale": args.prediction_radius_scale,
                    "bottom_boost": args.bottom_boost,
                    "bottom_boost_start": args.bottom_boost_start,
                    "bottom_boost_gate": args.bottom_boost_gate,
                    "low_okta_base_cf_max": args.low_okta_base_cf_max,
                    "low_okta_cap_okta": args.low_okta_cap_okta,
                },
                "tuning_history": tuning_history,
            }
            save_metadata(metadata_path, metadata)

            if capture_hyta_result is not None:
                append_hyta_prediction_csv(
                    base_dir / "hyta_predictions.csv",
                    {
                        "image_id": f"{index:04d}",
                        "timestamp": metadata["timestamp"],
                        "raw_file": str(image_path),
                        "processed_image": (
                            "" if processed_image_path is None else str(processed_image_path)
                        ),
                        "hyta_debug_image": (
                            str(sample_dir / "captured_hyta_debug.png")
                            if args.save_hyta_debug_image or args.debug_hyta
                            else ""
                        ),
                        "cloud_percent": f"{capture_hyta_result['cloud_percent']:.6f}",
                        "cloud_fraction": f"{capture_hyta_result['cloud_fraction']:.8f}",
                        "okta": capture_hyta_result["okta"],
                        "okta_text": capture_hyta_result["okta_text"],
                        "threshold": f"{capture_hyta_result['threshold']:.8f}",
                        "sigma": f"{capture_hyta_result['sigma']:.8f}",
                        "hyta_method": capture_hyta_result["method"],
                        "bottom_boost": capture_hyta_result["bottom_boost"],
                        "bottom_boost_start": capture_hyta_result["bottom_boost_start"],
                        "bottom_boost_gate": capture_hyta_result["bottom_boost_gate"],
                        "low_okta_base_cf_max": capture_hyta_result["low_okta_base_cf_max"],
                        "low_okta_cap_okta": capture_hyta_result["low_okta_cap_okta"],
                        "exposure": final_controls.get("exposure", active_controls["exposure"]),
                        "analogue_gain": final_controls.get(
                            "analogue_gain",
                            active_controls["analogue_gain"],
                        ),
                        "gain": final_controls.get("gain", active_controls["gain"]),
                        "ir_cut": active_filter,
                        "exposure_mode": profile["mode"],
                        "als_brightness_proxy": light_stats["brightness_proxy"],
                    },
                )
                if args.epaper_display:
                    try:
                        update_cloud_status_display(
                            cloud_percent=capture_hyta_result["cloud_percent"],
                            okta=capture_hyta_result["okta"],
                            prediction_time=prediction_time,
                            epaper_lib_path=args.epaper_lib_path,
                        )
                    except Exception as exc:
                        print(f"Could not update e-paper display: {exc}")

            index += 1
            if args.num_images == 0 or index < args.num_images:
                time.sleep(args.interval)
    finally:
        if preview_path.exists():
            try:
                preview_path.unlink()
            except OSError as exc:
                print(f"Could not remove temporary preview file {preview_path}: {exc}")
        if debug_server is not None:
            debug_server.shutdown()
            debug_server.server_close()
        if args.power_off_at_end:
            print("Powering off IMX385...")
            power_off_sensor()


if __name__ == "__main__":
    main()
