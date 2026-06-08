#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs

import cv2 as cv
import numpy as np


latest_jpeg: bytes | None = None
latest_lock = threading.Lock()
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ISP_TUNING_FILE = str(
    (
        SCRIPT_DIR.parent
        / "driver"
        / "imx385-libcamera-files"
        / "tuning_files"
        / "imx385_noalsc.json"
    ).resolve()
)


def detect_host_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "PI-IP"


def sanitize_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    cleaned = cleaned.strip("_")
    return cleaned or "capture"


def read_exact(stream, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def build_rpicam_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.rpicam_bin,
        "--camera",
        str(args.camera_index),
        "--mode",
        args.isp_sensor_mode,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--framerate",
        str(args.framerate),
        "--timeout",
        "0",
        "--codec",
        "yuv420",
        "--nopreview",
    ]
    if args.isp_tuning_file:
        cmd.extend(["--tuning-file", args.isp_tuning_file])
    if args.isp_awbgains:
        cmd.extend(["--awbgains", args.isp_awbgains])
    if args.isp_flicker_period_us is not None:
        cmd.extend(["--flicker-period", str(int(args.isp_flicker_period_us))])
    cmd.extend(["-o", "-"])
    return cmd


def build_rpicam_h264_cmd(args: argparse.Namespace, output_path: Path) -> list[str]:
    timeout_ms = int(args.duration * 1000) if args.duration > 0 else 0
    cmd = [
        args.rpicam_bin,
        "--camera",
        str(args.camera_index),
        "--mode",
        args.isp_sensor_mode,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--framerate",
        str(args.framerate),
        "--timeout",
        str(timeout_ms),
        "--codec",
        "h264",
        "--inline",
        "--nopreview",
    ]
    if args.h264_bitrate:
        cmd.extend(["--bitrate", str(args.h264_bitrate)])
    if args.isp_tuning_file:
        cmd.extend(["--tuning-file", args.isp_tuning_file])
    if args.isp_awbgains:
        cmd.extend(["--awbgains", args.isp_awbgains])
    if args.isp_flicker_period_us is not None:
        cmd.extend(["--flicker-period", str(int(args.isp_flicker_period_us))])
    cmd.extend(["-o", str(output_path)])
    return cmd


def build_rpicam_mjpeg_cmd(args: argparse.Namespace, output_path: Path) -> list[str]:
    timeout_ms = int(args.duration * 1000) if args.duration > 0 else 0
    cmd = [
        args.rpicam_bin,
        "--camera",
        str(args.camera_index),
        "--mode",
        args.isp_sensor_mode,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--framerate",
        str(args.framerate),
        "--timeout",
        str(timeout_ms),
        "--codec",
        "mjpeg",
        "--nopreview",
    ]
    if args.mjpeg_quality is not None:
        cmd.extend(["--quality", str(args.mjpeg_quality)])
    if args.isp_tuning_file:
        cmd.extend(["--tuning-file", args.isp_tuning_file])
    if args.isp_awbgains:
        cmd.extend(["--awbgains", args.isp_awbgains])
    if args.isp_flicker_period_us is not None:
        cmd.extend(["--flicker-period", str(int(args.isp_flicker_period_us))])
    cmd.extend(["-o", str(output_path)])
    return cmd


def build_ffmpeg_cmd(args: argparse.Namespace, output_path: Path) -> list[str]:
    return [
        args.ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        f"{args.width}x{args.height}",
        "-r",
        str(args.framerate),
        "-i",
        "-",
        "-c:v",
        args.ffmpeg_video_codec,
        "-preset",
        args.ffmpeg_preset,
        "-crf",
        str(args.ffmpeg_crf),
        str(output_path),
    ]


def build_remux_cmd(args: argparse.Namespace, h264_path: Path, mp4_path: Path) -> list[str]:
    return [
        args.ffmpeg_bin,
        "-y",
        "-framerate",
        str(args.framerate),
        "-i",
        str(h264_path),
        "-c:v",
        "copy",
        str(mp4_path),
    ]


def build_mjpeg_remux_cmd(args: argparse.Namespace, mjpeg_path: Path, avi_path: Path) -> list[str]:
    return [
        args.ffmpeg_bin,
        "-y",
        "-f",
        "mjpeg",
        "-framerate",
        str(args.framerate),
        "-i",
        str(mjpeg_path),
        "-c:v",
        "copy",
        str(avi_path),
    ]


def probe_video(args: argparse.Namespace, mp4_path: Path) -> dict:
    cmd = [
        args.ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,duration,nb_read_frames,nb_frames",
        "-of",
        "json",
        str(mp4_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError):
        return {}


def encode_preview(frame_bytes: bytes, width: int, height: int, preview_width: int, quality: int) -> bytes | None:
    yuv = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height * 3 // 2, width)
    frame = cv.cvtColor(yuv, cv.COLOR_YUV2BGR_I420)
    if preview_width and frame.shape[1] > preview_width:
        scale = preview_width / frame.shape[1]
        frame = cv.resize(
            frame,
            (preview_width, max(1, int(frame.shape[0] * scale))),
            interpolation=cv.INTER_AREA,
        )
    ok, encoded = cv.imencode(
        ".jpg",
        frame,
        [int(cv.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    return encoded.tobytes() if ok else None


class Mp4Recorder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.recording = False
        self.session_dir: Path | None = None
        self.mp4_path: Path | None = None
        self.metadata_path: Path | None = None
        self.timestamps_file = None
        self.timestamps_writer = None
        self.ffmpeg_proc: subprocess.Popen | None = None
        self.started_at_iso: str | None = None
        self.started_monotonic: float | None = None
        self.frame_count = 0
        self.last_error = ""
        self.last_saved_session = ""

    def start(self, base_name: str) -> tuple[bool, str]:
        with self.lock:
            if self.recording:
                return False, "Recording is already active."

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = sanitize_name(base_name)
            output_dir = Path(self.args.output_dir).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            self.session_dir = output_dir / f"{timestamp}_{safe_name}"
            self.session_dir.mkdir(parents=True, exist_ok=False)
            self.mp4_path = self.session_dir / "capture.mp4"
            self.metadata_path = self.session_dir / "metadata.json"

            timestamps_path = self.session_dir / "frame_timestamps.csv"
            self.timestamps_file = timestamps_path.open("w", newline="", encoding="utf-8")
            self.timestamps_writer = csv.writer(self.timestamps_file)
            self.timestamps_writer.writerow(["frame_index", "timestamp_iso", "seconds_since_start"])

            self.ffmpeg_proc = subprocess.Popen(
                build_ffmpeg_cmd(self.args, self.mp4_path),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.started_at_iso = datetime.now().isoformat(timespec="seconds")
            self.started_monotonic = time.monotonic()
            self.frame_count = 0
            self.last_error = ""
            self.recording = True
            return True, f"Recording started: {self.mp4_path}"

    def write_frame(self, frame_bytes: bytes) -> None:
        with self.lock:
            if not self.recording or self.ffmpeg_proc is None or self.ffmpeg_proc.stdin is None:
                return

            if self.ffmpeg_proc.poll() is not None:
                self.last_error = "ffmpeg stopped while recording"
                self._stop_locked()
                return

            timestamp_iso = datetime.now().isoformat(timespec="milliseconds")
            seconds_since_start = 0.0
            if self.started_monotonic is not None:
                seconds_since_start = time.monotonic() - self.started_monotonic

            try:
                self.ffmpeg_proc.stdin.write(frame_bytes)
                if self.timestamps_writer is not None:
                    self.timestamps_writer.writerow(
                        [self.frame_count, timestamp_iso, f"{seconds_since_start:.6f}"]
                    )
                self.frame_count += 1
            except (BrokenPipeError, OSError) as exc:
                self.last_error = f"Failed to write frame to ffmpeg: {exc}"
                self._stop_locked()

    def stop(self) -> tuple[bool, str]:
        with self.lock:
            if not self.recording:
                return False, "No recording is active."
            self._stop_locked()
            return True, f"Recording saved: {self.last_saved_session}"

    def _stop_locked(self) -> None:
        stopped_at_iso = datetime.now().isoformat(timespec="seconds")
        duration = 0.0
        if self.started_monotonic is not None:
            duration = max(time.monotonic() - self.started_monotonic, 0.0)

        if self.ffmpeg_proc is not None:
            if self.ffmpeg_proc.stdin is not None:
                try:
                    self.ffmpeg_proc.stdin.close()
                except OSError:
                    pass
            try:
                self.ffmpeg_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.ffmpeg_proc.terminate()
                try:
                    self.ffmpeg_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.ffmpeg_proc.kill()
                    self.ffmpeg_proc.wait()

        if self.timestamps_file is not None:
            self.timestamps_file.close()

        actual_fps = self.frame_count / duration if duration > 0 else 0.0
        if self.metadata_path is not None:
            metadata = {
                "camera_pipeline": "isp",
                "width": self.args.width,
                "height": self.args.height,
                "framerate_requested": self.args.framerate,
                "isp_sensor_mode": self.args.isp_sensor_mode,
                "isp_tuning_file": self.args.isp_tuning_file,
                "isp_awbgains": self.args.isp_awbgains,
                "isp_flicker_period_us": self.args.isp_flicker_period_us,
                "ffmpeg_video_codec": self.args.ffmpeg_video_codec,
                "ffmpeg_preset": self.args.ffmpeg_preset,
                "ffmpeg_crf": self.args.ffmpeg_crf,
                "started_at": self.started_at_iso,
                "stopped_at": stopped_at_iso,
                "duration_seconds": round(duration, 6),
                "frame_count": self.frame_count,
                "estimated_actual_fps": round(actual_fps, 6),
                "mp4_file": self.mp4_path.name if self.mp4_path else None,
            }
            self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        self.last_saved_session = str(self.mp4_path) if self.mp4_path else ""
        self.recording = False
        self.ffmpeg_proc = None
        self.timestamps_file = None
        self.timestamps_writer = None

    def snapshot(self) -> dict:
        with self.lock:
            duration = 0.0
            if self.recording and self.started_monotonic is not None:
                duration = time.monotonic() - self.started_monotonic
            fps = self.frame_count / duration if duration > 0 else 0.0
            return {
                "recording": self.recording,
                "session_dir": str(self.session_dir) if self.session_dir else "",
                "mp4_path": str(self.mp4_path) if self.mp4_path else "",
                "frame_count": self.frame_count,
                "duration_seconds": round(duration, 2),
                "estimated_capture_fps": round(fps, 2),
                "last_error": self.last_error,
                "last_saved_session": self.last_saved_session,
            }


class CameraWorker:
    def __init__(self, args: argparse.Namespace, recorder: Mp4Recorder, publish_preview: bool) -> None:
        self.args = args
        self.recorder = recorder
        self.publish_preview = publish_preview
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.frame_count = 0
        self.last_frame_time = 0.0
        self.last_error = ""
        self.last_command = ""
        self.stderr_tail: list[str] = []

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.thread is not None:
            self.thread.join(timeout=2)

    def snapshot(self) -> dict:
        with self.lock:
            age = time.monotonic() - self.last_frame_time if self.last_frame_time else None
            return {
                "frame_count": self.frame_count,
                "seconds_since_last_frame": None if age is None else round(age, 2),
                "last_error": self.last_error,
                "last_command": self.last_command,
                "stderr_tail": list(self.stderr_tail[-8:]),
            }

    def _remember_stderr(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        with self.lock:
            self.stderr_tail.append(line)
            self.stderr_tail = self.stderr_tail[-20:]

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        if proc.stderr is None:
            return
        for raw_line in proc.stderr:
            line = raw_line.decode("utf-8", errors="replace")
            self._remember_stderr(line)
            if "ERROR" in line or "Failed" in line or "no cameras" in line:
                with self.lock:
                    self.last_error = line.strip()

    def _run(self) -> None:
        global latest_jpeg

        frame_size = self.args.width * self.args.height * 3 // 2
        frame_index = 0

        while not self.stop_event.is_set():
            cmd = build_rpicam_cmd(self.args)
            command_text = " ".join(cmd)
            with self.lock:
                self.last_command = command_text
                self.last_error = ""
                self.stderr_tail = []
            print("Starting ISP camera:", command_text, flush=True)
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stderr_thread = threading.Thread(target=self._read_stderr, args=(self.proc,), daemon=True)
            stderr_thread.start()

            try:
                assert self.proc.stdout is not None
                while not self.stop_event.is_set():
                    frame_bytes = read_exact(self.proc.stdout, frame_size)
                    if frame_bytes is None:
                        if self.proc.poll() is not None:
                            with self.lock:
                                if not self.last_error:
                                    self.last_error = f"rpicam stopped with exit code {self.proc.returncode}"
                        break

                    self.recorder.write_frame(frame_bytes)
                    with self.lock:
                        self.frame_count += 1
                        self.last_frame_time = time.monotonic()

                    if self.publish_preview and frame_index % max(1, self.args.preview_every) == 0:
                        jpeg = encode_preview(
                            frame_bytes,
                            self.args.width,
                            self.args.height,
                            self.args.preview_width,
                            self.args.jpeg_quality,
                        )
                        if jpeg is not None:
                            with latest_lock:
                                latest_jpeg = jpeg
                    frame_index += 1
            finally:
                if self.proc is not None and self.proc.poll() is None:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                        self.proc.wait()
                self.proc = None

            if not self.stop_event.is_set():
                print("ISP camera stopped, restarting...", flush=True)
                time.sleep(0.5)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def render_home_page(recorder: Mp4Recorder, camera: CameraWorker | None = None) -> str:
    status = recorder.snapshot()
    return f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>IMX385 Recorder</title>
      <style>
        body {{ margin: 0; background: #111; color: #eee; font-family: Arial, sans-serif; }}
        .page {{ max-width: 1180px; margin: 0 auto; padding: 18px; }}
        img {{ width: 100%; max-width: 1100px; display: block; background: #000; }}
        .controls {{ display: flex; gap: 12px; align-items: end; margin: 16px 0; flex-wrap: wrap; }}
        label {{ display: grid; gap: 4px; }}
        input {{ padding: 8px; min-width: 220px; }}
        button {{ padding: 9px 14px; cursor: pointer; }}
        .status {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px 16px; }}
      </style>
    </head>
    <body>
      <div class="page">
        <h2>IMX385 ISP recorder</h2>
        <img src="/stream.mjpg" />
        <div class="controls">
          <label>
            Session name
            <input id="sessionName" value="turbine_test" />
          </label>
          <button onclick="startRecording()">Start recording</button>
          <button onclick="stopRecording()">Stop recording</button>
        </div>
        <div class="status" id="status">
          <div><strong>Video length:</strong> {status["duration_seconds"]:.1f} s</div>
          <div><strong>Last saved video:</strong> {status["last_saved_session"] or "-"}</div>
          <div><strong>Last error:</strong> {status["last_error"] or "None"}</div>
        </div>
      </div>
      <script>
        async function refreshStatus() {{
          const response = await fetch('/status');
          const data = await response.json();
          document.getElementById('status').innerHTML = `
            <div><strong>Video length:</strong> ${{Number(data.duration_seconds || 0).toFixed(1)}} s</div>
            <div><strong>Last saved video:</strong> ${{data.last_saved_session || '-'}}</div>
            <div><strong>Last error:</strong> ${{data.last_error || 'None'}}</div>
          `;
        }}
        async function startRecording() {{
          const params = new URLSearchParams();
          params.set('session_name', document.getElementById('sessionName').value);
          await fetch('/start', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/x-www-form-urlencoded' }},
            body: params.toString()
          }});
          refreshStatus();
        }}
        async function stopRecording() {{
          await fetch('/stop', {{ method: 'POST' }});
          refreshStatus();
        }}
        refreshStatus();
        setInterval(refreshStatus, 1000);
      </script>
    </body>
    </html>
    """


class RecorderHandler(BaseHTTPRequestHandler):
    recorder: Mp4Recorder | None = None
    camera: CameraWorker | None = None

    def log_message(self, format: str, *args) -> None:
        return

    def _send(self, status_code: int, body: str, content_type: str = "text/plain; charset=utf-8") -> None:
        encoded = body.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        global latest_jpeg

        if self.recorder is None:
            self._send(500, "Recorder not configured")
            return

        if self.path == "/":
            self._send(200, render_home_page(self.recorder, self.camera), "text/html; charset=utf-8")
            return

        if self.path == "/status":
            status = self.recorder.snapshot()
            if self.camera is not None:
                status["camera"] = self.camera.snapshot()
            self._send(200, json.dumps(status), "application/json; charset=utf-8")
            return

        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            while True:
                with latest_lock:
                    frame = latest_jpeg
                if frame is None:
                    time.sleep(0.05)
                    continue
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        if self.recorder is None:
            self._send(500, "Recorder not configured")
            return

        if self.path == "/start":
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length).decode("utf-8")
            form = parse_qs(raw)
            session_name = form.get("session_name", ["turbine_test"])[0]
            ok, message = self.recorder.start(session_name)
            self._send(200 if ok else 409, message)
            return

        if self.path == "/stop":
            ok, message = self.recorder.stop()
            self._send(200 if ok else 409, message)
            return

        self.send_response(404)
        self.end_headers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record IMX385 ISP video as MP4")
    parser.add_argument("--http", action="store_true", help="Start web preview with start/stop recording buttons")
    parser.add_argument("--port", type=int, default=8094, help="HTTP port when using --http")
    parser.add_argument("--duration", type=float, default=0.0, help="Recording duration in seconds. 0 means until Ctrl+C.")
    parser.add_argument("--base-name", default="turbine_test", help="Name suffix for the recording folder")
    parser.add_argument("--output-dir", default="recordings", help="Directory for recording folders")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--framerate", type=float, default=30.0)
    parser.add_argument("--isp-sensor-mode", default="1952:1080:10:U")
    parser.add_argument("--rpicam-bin", default="rpicam-vid")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--isp-tuning-file", default=DEFAULT_ISP_TUNING_FILE)
    parser.add_argument("--isp-awbgains", default=None)
    parser.add_argument("--isp-flicker-period-us", type=int, default=None)
    parser.add_argument(
        "--recording-backend",
        choices=("direct-mjpeg", "direct-h264", "python-yuv"),
        default="direct-mjpeg",
        help="Backend for non-HTTP recording. direct-mjpeg avoids Python raw-frame encoding.",
    )
    parser.add_argument("--h264-bitrate", type=int, default=12000000, help="Bitrate for direct H.264 recording")
    parser.add_argument("--mjpeg-quality", type=int, default=90, help="JPEG quality for direct MJPEG recording")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--ffmpeg-video-codec", default="libx264")
    parser.add_argument("--ffmpeg-preset", default="ultrafast")
    parser.add_argument("--ffmpeg-crf", type=int, default=23)
    parser.add_argument("--preview-width", type=int, default=1280)
    parser.add_argument("--preview-every", type=int, default=3)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if args.width % 2 or args.height % 2:
        raise ValueError("--width and --height must be even for YUV420")
    if args.framerate <= 0:
        raise ValueError("--framerate must be positive")
    if args.duration < 0:
        raise ValueError("--duration must be 0 or greater")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if args.preview_every < 1:
        raise ValueError("--preview-every must be at least 1")


def run_http(args: argparse.Namespace) -> None:
    recorder = Mp4Recorder(args)
    camera = CameraWorker(args, recorder, publish_preview=True)
    camera.start()

    RecorderHandler.recorder = recorder
    RecorderHandler.camera = camera
    server = ThreadedHTTPServer(("0.0.0.0", args.port), RecorderHandler)

    print(f"Recorder preview: http://{detect_host_ip()}:{args.port}/")
    print("Camera pipeline: isp/libcamera")
    print(f"ISP sensor mode: {args.isp_sensor_mode}")
    print(f"Recording output dir: {Path(args.output_dir).expanduser().resolve()}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()
        camera.stop()
        server.shutdown()
        server.server_close()


def run_terminal(args: argparse.Namespace) -> None:
    if args.recording_backend == "direct-h264":
        run_terminal_direct_h264(args)
    elif args.recording_backend == "direct-mjpeg":
        run_terminal_direct_mjpeg(args)
    else:
        run_terminal_python_yuv(args)


def run_terminal_python_yuv(args: argparse.Namespace) -> None:
    recorder = Mp4Recorder(args)
    camera = CameraWorker(args, recorder, publish_preview=False)

    ok, message = recorder.start(args.base_name)
    if not ok:
        raise RuntimeError(message)
    print(message)
    print("Camera pipeline: isp/libcamera")
    print(f"ISP sensor mode: {args.isp_sensor_mode}")
    if args.duration > 0:
        print(f"Recording for {args.duration:.1f} seconds...")
    else:
        print("Recording until Ctrl+C...")

    camera.start()
    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping recording...")
    finally:
        ok, message = recorder.stop()
        camera.stop()
        print(message)


def run_terminal_direct_h264(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = sanitize_name(args.base_name)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    session_dir = output_dir / f"{timestamp}_{safe_name}"
    session_dir.mkdir(parents=True, exist_ok=False)

    h264_path = session_dir / "capture.h264"
    mp4_path = session_dir / "capture.mp4"
    metadata_path = session_dir / "metadata.json"

    cmd = build_rpicam_h264_cmd(args, h264_path)
    started_at_iso = datetime.now().isoformat(timespec="seconds")
    started_monotonic = time.monotonic()

    print("Starting direct H.264 recording:")
    print(" ".join(cmd))
    print("Camera pipeline: isp/libcamera")
    print(f"ISP sensor mode: {args.isp_sensor_mode}")
    print(f"Recording session: {session_dir}")
    if args.duration > 0:
        print(f"Recording for {args.duration:.1f} seconds...")
    else:
        print("Recording until Ctrl+C...")

    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping recording...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    stopped_at_iso = datetime.now().isoformat(timespec="seconds")
    duration = max(time.monotonic() - started_monotonic, 0.0)

    h264_size = h264_path.stat().st_size if h264_path.exists() else 0
    if proc.returncode not in (0, -15) and h264_size == 0:
        raise RuntimeError(f"rpicam-vid failed with exit code {proc.returncode}")

    remux_cmd = build_remux_cmd(args, h264_path, mp4_path)
    print("Remuxing to MP4:")
    print(" ".join(remux_cmd))
    subprocess.run(remux_cmd, check=True)

    probe = probe_video(args, mp4_path)
    metadata = {
        "camera_pipeline": "isp",
        "recording_backend": "direct-h264",
        "width": args.width,
        "height": args.height,
        "framerate_requested": args.framerate,
        "isp_sensor_mode": args.isp_sensor_mode,
        "isp_tuning_file": args.isp_tuning_file,
        "isp_awbgains": args.isp_awbgains,
        "isp_flicker_period_us": args.isp_flicker_period_us,
        "h264_bitrate": args.h264_bitrate,
        "started_at": started_at_iso,
        "stopped_at": stopped_at_iso,
        "duration_seconds": round(duration, 6),
        "h264_file": h264_path.name,
        "mp4_file": mp4_path.name,
        "ffprobe": probe,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved H.264: {h264_path}")
    print(f"Saved MP4: {mp4_path}")
    print(f"Saved metadata: {metadata_path}")


def run_terminal_direct_mjpeg(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = sanitize_name(args.base_name)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    session_dir = output_dir / f"{timestamp}_{safe_name}"
    session_dir.mkdir(parents=True, exist_ok=False)

    mjpeg_path = session_dir / "capture.mjpeg"
    avi_path = session_dir / "capture.avi"
    metadata_path = session_dir / "metadata.json"

    cmd = build_rpicam_mjpeg_cmd(args, mjpeg_path)
    started_at_iso = datetime.now().isoformat(timespec="seconds")
    started_monotonic = time.monotonic()

    print("Starting direct MJPEG recording:")
    print(" ".join(cmd))
    print("Camera pipeline: isp/libcamera")
    print(f"ISP sensor mode: {args.isp_sensor_mode}")
    print(f"Recording session: {session_dir}")
    if args.duration > 0:
        print(f"Recording for {args.duration:.1f} seconds...")
    else:
        print("Recording until Ctrl+C...")

    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping recording...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    stopped_at_iso = datetime.now().isoformat(timespec="seconds")
    duration = max(time.monotonic() - started_monotonic, 0.0)
    mjpeg_size = mjpeg_path.stat().st_size if mjpeg_path.exists() else 0
    if proc.returncode not in (0, -15) and mjpeg_size == 0:
        raise RuntimeError(f"rpicam-vid failed with exit code {proc.returncode}")

    remux_cmd = build_mjpeg_remux_cmd(args, mjpeg_path, avi_path)
    print("Remuxing to AVI:")
    print(" ".join(remux_cmd))
    subprocess.run(remux_cmd, check=True)

    probe = probe_video(args, avi_path)
    metadata = {
        "camera_pipeline": "isp",
        "recording_backend": "direct-mjpeg",
        "width": args.width,
        "height": args.height,
        "framerate_requested": args.framerate,
        "isp_sensor_mode": args.isp_sensor_mode,
        "isp_tuning_file": args.isp_tuning_file,
        "isp_awbgains": args.isp_awbgains,
        "isp_flicker_period_us": args.isp_flicker_period_us,
        "mjpeg_quality": args.mjpeg_quality,
        "started_at": started_at_iso,
        "stopped_at": stopped_at_iso,
        "duration_seconds": round(duration, 6),
        "mjpeg_file": mjpeg_path.name,
        "avi_file": avi_path.name,
        "ffprobe": probe,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved MJPEG: {mjpeg_path}")
    print(f"Saved AVI: {avi_path}")
    print(f"Saved metadata: {metadata_path}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.http:
        run_http(args)
    else:
        run_terminal(args)


if __name__ == "__main__":
    main()
