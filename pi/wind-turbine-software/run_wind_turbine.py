import argparse
import csv
import socket
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

import cv2 as cv
import numpy as np

from rpm import bpm_cascade
from rpm import opticalflow
from rpm import utils
from epaper_status import DEFAULT_EPAPER_LIB_PATH, update_rpm_status_display


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
DEFAULT_RESULTS_PATH = str((SCRIPT_DIR / "runs" / "wind_turbine_rpm.csv").resolve())
DEFAULT_CONFIG_PATH = str((SCRIPT_DIR / "config" / "rye_config.json").resolve())
latest_jpeg = None


class RealtimePacer:
    def __init__(self, enabled, fps):
        self.enabled = bool(enabled)
        self.frame_period = 1.0 / float(fps) if fps and fps > 0 else 0.0
        self.next_frame_time = time.perf_counter()

    def wait(self):
        if not self.enabled or self.frame_period <= 0:
            return

        self.next_frame_time += self.frame_period
        delay = self.next_frame_time - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self.next_frame_time = time.perf_counter()


class EpaperRpmUpdater:
    def __init__(self, enabled, interval, epaper_lib_path):
        self.enabled = bool(enabled)
        self.interval = float(interval)
        self.epaper_lib_path = epaper_lib_path
        self.last_update_time = 0.0

    def update(self, rpm):
        if not self.enabled:
            return

        now = time.monotonic()
        if self.last_update_time and now - self.last_update_time < self.interval:
            return

        try:
            update_rpm_status_display(
                rpm=rpm,
                update_time=datetime.now(),
                epaper_lib_path=self.epaper_lib_path,
            )
            self.last_update_time = now
        except Exception as exc:
            print(f"Could not update e-paper display: {exc}", flush=True)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class PreviewHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        global latest_jpeg

        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"""
                <html>
                <head><title>Wind turbine RPM preview</title></head>
                <body style="margin:0;background:#111;color:white;font-family:sans-serif;">
                <div style="padding:10px;">Wind turbine RPM preview</div>
                <img src="/stream.mjpg" style="width:100%;max-width:1100px;display:block;margin:auto;" />
                </body>
                </html>
                """
            )
            return

        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            while True:
                if latest_jpeg is None:
                    time.sleep(0.05)
                    continue

                self.wfile.write(b"--frame\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(latest_jpeg)))
                self.end_headers()
                self.wfile.write(latest_jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(1 / 30)
            return

        self.send_response(404)
        self.end_headers()


def detect_host_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "PI-IP"


def start_http_preview(port):
    server = ThreadedHTTPServer(("0.0.0.0", port), PreviewHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"HTTP preview: http://{detect_host_ip()}:{port}/")
    return server


def publish_preview(frame, lines, width=640, jpeg_quality=65):
    global latest_jpeg
    preview = frame.copy()
    if width and preview.shape[1] > width:
        scale = width / preview.shape[1]
        preview = cv.resize(
            preview,
            (width, max(1, int(preview.shape[0] * scale))),
            interpolation=cv.INTER_AREA,
        )

    y = 30
    for line in lines:
        cv.putText(
            preview,
            line,
            (20, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
            cv.LINE_AA,
        )
        cv.putText(
            preview,
            line,
            (20, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )
        y += 32

    ok, jpeg = cv.imencode(
        ".jpg",
        preview,
        [int(cv.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if ok:
        latest_jpeg = jpeg.tobytes()


def build_video_output_path(out_path):
    return out_path.with_suffix(".mp4")


def ensure_video_writer(video_writer, frame, fps, output_path):
    if video_writer is not None:
        return video_writer

    height, width = frame.shape[:2]
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(
        str(output_path),
        fourcc,
        float(fps),
        (width, height),
    )
    return writer


def resolve_video_target(target, config_path):
    target_path = Path(target).expanduser()
    if target_path.is_absolute():
        return str(target_path)

    cwd_path = (Path.cwd() / target_path).resolve()
    if cwd_path.exists():
        return str(cwd_path)

    config_relative_path = (config_path.parent / target_path).resolve()
    if config_relative_path.exists():
        return str(config_relative_path)

    raise FileNotFoundError(
        "Could not find video file. Tried:\n"
        f"  - {cwd_path}\n"
        f"  - {config_relative_path}\n"
        "Use --video-path with the full path to the video file."
    )


def apply_cli_overrides(params, args, config_path):
    params["source_type"] = args.source
    if args.mode is not None:
        params["mode"] = args.mode
    if args.fps is not None:
        params["fps"] = args.fps
    if args.loop_video is not None:
        params["loop_video"] = args.loop_video

    if args.source == "video":
        if args.video_path is not None:
            params["target"] = args.video_path
        if not params.get("target"):
            raise ValueError("Video source needs --video-path or target in config.")
        params["target"] = resolve_video_target(params["target"], config_path)
    else:
        params["width"] = args.width
        params["height"] = args.height
        params["isp_sensor_mode"] = args.isp_sensor_mode
        params["rpicam_bin"] = args.rpicam_bin
        params["camera_index"] = args.camera_index
        params["isp_tuning_file"] = args.isp_tuning_file
        params["isp_awbgains"] = args.isp_awbgains
        params["camera_rotation"] = args.camera_rotation
        params["camera_frame_timeout"] = args.camera_frame_timeout

    return params


def should_visualize(args):
    return args.display or args.http_preview or args.save_video


def run_optical_flow(feed, params, args, writer, video_writer, video_output_path, epaper_updater):
    rpms = []
    errors = []
    visualize = should_visualize(args)
    pacer = RealtimePacer(args.realtime, params["fps"])
    while True:
        if feed.isActive:
            data, image = feed.get_optical_flow_vectors()
            if image is None:
                continue

            if all(x is not None for x in data):
                motion_vectors = data[0] - data[1]
                scaled_vectors = motion_vectors * feed.rpm_scaling_factor
                rpm = feed.calculate_rpm_from_vectors(scaled_vectors)
                flow_image = feed.draw_optical_flow(image, data[1], data[0]) if visualize else image
            else:
                rpm = None
                flow_image = image

            if rpm is not None:
                rpms.append(rpm)
                error = utils.calculate_error_percentage(rpm, params["real_rpm"])
                errors.append(error)

            if args.save_video:
                video_writer = ensure_video_writer(
                    video_writer,
                    flow_image,
                    params["fps"],
                    video_output_path,
                )
                video_writer.write(flow_image)

            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "frame": feed.frame_cnt,
                    "mode": "opticalflow",
                    "rpm": "" if rpm is None else round(float(rpm), 4),
                    "smoothed_rpm": "" if rpm is None else round(float(rpm), 4),
                    "delta": "",
                    "threshold": "",
                    "mode_value": "",
                }
            )
            epaper_updater.update(rpm)

            if args.http_preview and feed.frame_cnt % args.preview_every == 0:
                publish_preview(
                    flow_image,
                    [
                        f"Frame: {feed.frame_cnt}",
                        f"RPM: {'' if rpm is None else round(float(rpm), 3)}",
                    ],
                    width=args.preview_width,
                    jpeg_quality=args.jpeg_quality,
                )

            if args.display:
                cv.imshow("Wind turbine RPM feed", flow_image)
                k = cv.waitKey(1) & 0xFF
                if k == 27:
                    break

            if args.max_frames and feed.frame_cnt >= args.max_frames:
                break
            pacer.wait()
        else:
            break

    return video_writer


def run_bpm(feed, params, args, writer, video_writer, video_output_path, epaper_updater):
    frame = feed.get_frame()
    box_params = feed.get_fitted_box_params_from_cfg()
    bounds = feed.cascade_bounding_boxes(*box_params)
    kernel_er_dil_params = feed.get_dilation_erosion_params()
    visualize = should_visualize(args)

    frame_ticks = deque(maxlen=2)
    fb_average_long_buffer = deque(maxlen=int(params["fps"] * 60))
    rpm_buffer = deque(maxlen=params["rpm_buffer_length"])
    deviation, mode = 0, 0
    prev_rpm, rpm = 0, 0
    had_rpm_measurement = False
    rpm_zero_timeout_frames = max(
        1,
        int(round(params["fps"] * params.get("rpm_zero_timeout_seconds", 10.0))),
    )
    feed.process_rpm_bounds()
    pacer = RealtimePacer(args.realtime, params["fps"])

    while True:
        if feed.isActive:
            display_frame = frame.copy() if visualize else None
            for bounding_box in bounds.values():
                processed_region = bounding_box.dilate_and_erode(
                    frame, *kernel_er_dil_params
                )
                bounding_box.fb.insert(processed_region)

                if visualize:
                    bounding_box.draw.border_around_region(
                        processed_region, 1, [0, 255, 0]
                    )
                    display_frame = bounding_box.draw.processing_results(
                        display_frame, bounding_box.region, processed_region
                    )

                bounding_box.fb.update_color_delta_average()

            if feed.frame_cnt % feed.color_delta_update_frequency == 0:
                feed.update_global_fb_average()
                fb_average_long_buffer.append(feed.all_fb_delta_average)
                mode = utils.find_top_n_modes(fb_average_long_buffer, 1)
                mode = np.mean(mode)
                deviation = np.std(fb_average_long_buffer)

            detected = feed.blade_detection_in_box_regions(float(deviation), float(mode))
            if detected:
                frame_ticks.append(feed.frame_cnt)
                if len(frame_ticks) == 2:
                    rpm = feed.calculate_rpm(frame_ticks[1] - frame_ticks[0], feed.fps)
                    if rpm_buffer:
                        if feed.rpm_within_bounds(rpm, prev_rpm):
                            rpm_buffer.append(rpm)
                            had_rpm_measurement = True
                    else:
                        rpm_buffer.append((rpm if rpm < 30 else 0))
                        had_rpm_measurement = True
                feed.detection_enable_toggle = False

            feed.update_detection_enable_toggle(
                feed.all_fb_delta_average, deviation, mode, frame_ticks
            )

            rpm_is_stale = (
                had_rpm_measurement
                and frame_ticks
                and feed.frame_cnt - frame_ticks[-1] > rpm_zero_timeout_frames
            )
            if rpm_is_stale:
                rpm_buffer.clear()
                rpm = 0
                prev_rpm = 0

            smoothed_rpm = (
                0.0
                if rpm_is_stale
                else "" if not rpm_buffer else round(float(np.mean(rpm_buffer)), 3)
            )
            #print("RPM buffer:", list(rpm_buffer), "Smoothed RPM:", smoothed_rpm)
            threshold = mode + feed.threshold_multiplier * deviation
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "frame": feed.frame_cnt,
                    "mode": "bpm",
                    "rpm": "" if not rpm else round(float(rpm), 4),
                    "smoothed_rpm": smoothed_rpm,
                    "delta": round(float(feed.all_fb_delta_average), 4),
                    "threshold": round(float(threshold), 4),
                    "mode_value": round(float(mode), 4),
                }
            )
            epaper_updater.update(smoothed_rpm if smoothed_rpm != "" else rpm)

            if args.save_video:
                video_writer = ensure_video_writer(
                    video_writer,
                    display_frame,
                    params["fps"],
                    video_output_path,
                )
                video_writer.write(display_frame)

            if args.http_preview and feed.frame_cnt % args.preview_every == 0:
                publish_preview(
                    display_frame,
                    [
                        f"Frame: {feed.frame_cnt}",
                        f"RPM: {'' if not rpm else round(float(rpm), 3)}",
                        f"Smoothed RPM: {smoothed_rpm}",
                    ],
                    width=args.preview_width,
                    jpeg_quality=args.jpeg_quality,
                )

            if args.display:
                cv.imshow("Wind turbine RPM feed", display_frame)
                k = cv.waitKey(1) & 0xFF
                if k == 27:
                    break

            if args.max_frames and feed.frame_cnt >= args.max_frames:
                break

            pacer.wait()
            frame = feed.get_frame()
            prev_rpm = rpm
        else:
            break

    return video_writer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source", choices=["video", "camera"], default="camera")
    parser.add_argument("--mode", choices=["bpm", "opticalflow"], default="bpm")
    parser.add_argument("--video-path", default=None)
    parser.add_argument(
        "--loop-video",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--out", default=DEFAULT_RESULTS_PATH)
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--http-preview", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--http-port", type=int, default=8092)
    parser.add_argument("--preview-every", type=int, default=3)
    parser.add_argument("--preview-width", type=int, default=640)
    parser.add_argument("--jpeg-quality", type=int, default=65)
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Throttle prerecorded video processing to --fps for real-time preview.",
    )
    parser.add_argument(
        "--results-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable HTTP preview, local display, and saved video so only CSV results are written.",
    )
    parser.add_argument(
        "--epaper-display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update the e-paper display with RPM status.",
    )
    parser.add_argument(
        "--epaper-interval",
        type=float,
        default=120.0,
        help="Seconds between e-paper RPM display updates.",
    )
    parser.add_argument(
        "--epaper-lib-path",
        default=DEFAULT_EPAPER_LIB_PATH,
        help="Path to the Waveshare e-paper Python library.",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--isp-sensor-mode", default="1952:1080:10:U")
    parser.add_argument("--rpicam-bin", default="rpicam-vid")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--camera-rotation",
        type=int,
        choices=[0, 90, 180, 270],
        default=180,
        help="Rotate ISP camera frames before RPM processing and preview.",
    )
    parser.add_argument(
        "--isp-tuning-file",
        default=DEFAULT_ISP_TUNING_FILE,
        help="Optional libcamera tuning file for ISP camera mode.",
    )
    parser.add_argument(
        "--isp-awbgains",
        default=None,
        help="Optional fixed ISP AWB gains, for example 1.4,1.8.",
    )
    parser.add_argument(
        "--camera-frame-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for the first/next camera frame before printing rpicam diagnostics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.results_only:
        args.http_preview = False
        args.display = False
        args.save_video = False

    config_path = Path(args.config).expanduser().resolve()
    params = apply_cli_overrides(utils.parse_json(str(config_path)), args, config_path)
    if params["source_type"] == "camera":
        print("Camera pipeline: isp/libcamera")
        print(f"ISP sensor mode: {params.get('isp_sensor_mode')}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    video_output_path = build_video_output_path(out_path)
    video_writer = None
    epaper_updater = EpaperRpmUpdater(
        enabled=args.epaper_display,
        interval=args.epaper_interval,
        epaper_lib_path=args.epaper_lib_path,
    )

    if params["mode"] == "bpm":
        feed = bpm_cascade.BpmCascade(**params)
    else:
        feed = opticalflow.OpticalFlow(**params)

    preview_server = start_http_preview(args.http_port) if args.http_preview else None

    fieldnames = [
        "timestamp",
        "frame",
        "mode",
        "rpm",
        "smoothed_rpm",
        "delta",
        "threshold",
        "mode_value",
    ]

    try:
        with out_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            if params["mode"] == "bpm":
                video_writer = run_bpm(
                    feed,
                    params,
                    args,
                    writer,
                    video_writer,
                    video_output_path,
                    epaper_updater,
                )
            else:
                video_writer = run_optical_flow(
                    feed,
                    params,
                    args,
                    writer,
                    video_writer,
                    video_output_path,
                    epaper_updater,
                )
    finally:
        if hasattr(feed, "video") and hasattr(feed.video, "release"):
            feed.video.release()
        if video_writer is not None:
            video_writer.release()
        if preview_server is not None:
            preview_server.shutdown()
            preview_server.server_close()
        if args.display:
            cv.destroyAllWindows()

    print(f"Saved RPM results to {out_path}")
    if args.save_video:
        print(f"Saved video feed to {video_output_path}")


if __name__ == "__main__":
    main()
