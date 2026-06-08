import subprocess
import threading
import select

import cv2 as cv
import numpy as np


class OpenCvSource:
    def __init__(self, target, loop_video=False):
        self.target = str(target)
        self.loop_video = bool(loop_video)
        self.capture = cv.VideoCapture(self.target)

    def read(self):
        ok, frame = self.capture.read()
        if ok or not self.loop_video:
            return ok, frame

        self.capture.set(cv.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.capture.read()
        if ok:
            return ok, frame

        self.capture.release()
        self.capture = cv.VideoCapture(self.target)
        return self.capture.read()

    def release(self):
        self.capture.release()


class Imx385IspSource:
    def __init__(
        self,
        width=1920,
        height=1080,
        sensor_mode="1952:1080:10:U",
        framerate=30,
        rpicam_bin="rpicam-vid",
        camera_index=0,
        tuning_file=None,
        awbgains=None,
        rotation=0,
        frame_timeout=5.0,
    ):
        self.width = int(width)
        self.height = int(height)
        self.sensor_mode = sensor_mode
        self.framerate = float(framerate)
        self.rpicam_bin = rpicam_bin
        self.camera_index = int(camera_index)
        self.tuning_file = tuning_file
        self.awbgains = awbgains
        self.rotation = int(rotation)
        self.frame_timeout = float(frame_timeout)
        self.frame_size = self.width * self.height * 3 // 2
        self.process = None
        self.stderr_tail = []
        self.last_error = ""
        self.command = []

    def _remember_stderr(self, line):
        line = line.strip()
        if not line:
            return
        self.stderr_tail.append(line)
        self.stderr_tail = self.stderr_tail[-20:]
        if "ERROR" in line or "Failed" in line or "no cameras" in line:
            self.last_error = line

    def _start_stream(self):
        if self.process is not None and self.process.poll() is None:
            return True

        self.release()
        cmd = [
            self.rpicam_bin,
            "--camera",
            str(self.camera_index),
            "--mode",
            self.sensor_mode,
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--framerate",
            str(self.framerate),
            "--timeout",
            "0",
            "--codec",
            "yuv420",
            "--nopreview",
        ]
        if self.tuning_file:
            cmd.extend(["--tuning-file", self.tuning_file])
        if self.awbgains:
            cmd.extend(["--awbgains", self.awbgains])
        cmd.extend(["-o", "-"])
        self.command = cmd
        print("Starting ISP camera:", " ".join(cmd), flush=True)

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            threading.Thread(target=self._read_stderr, daemon=True).start()
        except OSError:
            self.process = None
            return False

        return True

    def _read_stderr(self):
        if self.process is None or self.process.stderr is None:
            return
        for raw_line in self.process.stderr:
            self._remember_stderr(raw_line.decode("utf-8", errors="replace"))

    def _read_exact_frame(self):
        if self.process is None or self.process.stdout is None:
            return None

        chunks = []
        remaining = self.frame_size
        while remaining > 0:
            ready, _, _ = select.select([self.process.stdout], [], [], self.frame_timeout)
            if not ready:
                self.last_error = f"Timed out waiting {self.frame_timeout:.1f}s for camera frame"
                return None
            chunk = self.process.stdout.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def read(self):
        if not self._start_stream():
            return False, None

        frame_bytes = self._read_exact_frame()
        if frame_bytes is None:
            message = self.last_error
            if self.process is not None and self.process.poll() is not None:
                message = message or f"rpicam stopped with exit code {self.process.returncode}"
            if not message:
                message = "Camera stream ended before a full frame was received"
            if self.stderr_tail:
                message += "\n" + "\n".join(self.stderr_tail[-8:])
            print(message, flush=True)
            self.release()
            return False, None

        yuv = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
            self.height * 3 // 2,
            self.width,
        )
        frame = cv.cvtColor(yuv, cv.COLOR_YUV2BGR_I420)
        if self.rotation == 90:
            frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            frame = cv.rotate(frame, cv.ROTATE_180)
        elif self.rotation == 270:
            frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)

        return True, frame

    def release(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None


def make_frame_source(params):
    source_type = params.get("source_type", "video")
    if source_type == "camera":
        return Imx385IspSource(
            width=params.get("width", 1920),
            height=params.get("height", 1080),
            sensor_mode=params.get("isp_sensor_mode", "1952:1080:10:U"),
            framerate=params.get("fps", 30),
            rpicam_bin=params.get("rpicam_bin", "rpicam-vid"),
            camera_index=params.get("camera_index", 0),
            tuning_file=params.get("isp_tuning_file"),
            awbgains=params.get("isp_awbgains"),
            rotation=params.get("camera_rotation", 0),
            frame_timeout=params.get("camera_frame_timeout", 5.0),
        )
    return OpenCvSource(
        params["target"],
        loop_video=params.get("loop_video", False),
    )
