#!/usr/bin/env python3

import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

BAYER_MAP = {
    "BGGR": cv2.COLOR_BayerBG2BGR,
    "RGGB": cv2.COLOR_BayerRG2BGR,
    "GRBG": cv2.COLOR_BayerGR2BGR,
    "GBRG": cv2.COLOR_BayerGB2BGR,
}


def unpack_raw10(chunk: bytes, h: int, w: int) -> np.ndarray:
    # Unpack RAW10 bytes to a H×W array of uint16
    b = np.frombuffer(chunk, dtype=np.uint8)
    b = b.reshape(-1, 5)

    # Extract 4 pixels per 5 bytes
    p0 = (b[:, 0].astype(np.uint16) << 2) | ((b[:, 4] >> 0) & 0b00000011)
    p1 = (b[:, 1].astype(np.uint16) << 2) | ((b[:, 4] >> 2) & 0b00000011)
    p2 = (b[:, 2].astype(np.uint16) << 2) | ((b[:, 4] >> 4) & 0b00000011)
    p3 = (b[:, 3].astype(np.uint16) << 2) | ((b[:, 4] >> 6) & 0b00000011)

    unpacked = np.empty((b.shape[0] * 4,), dtype=np.uint16)
    unpacked[0::4] = p0
    unpacked[1::4] = p1
    unpacked[2::4] = p2
    unpacked[3::4] = p3

    return unpacked.reshape((h, w))


def raw10_to_video(
    raw_path: str,
    width: int,
    height: int,
    bayer: str,
    fps: int,
    out_path: str,
):
    frame_bytes = int(width * height * 1.25)
    file_size = os.path.getsize(raw_path)
    n_frames = file_size // frame_bytes

    if file_size % frame_bytes:
        raise ValueError(
            f"File size {file_size} is not an exact multiple of one RAW10 frame"
            f"({frame_bytes}). Check width/height."
        )

    print(f"{raw_path}: {n_frames} frames detected ({width}×{height} RAW10)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    with open(raw_path, "rb") as f:
        for idx in range(n_frames):
            chunk = f.read(frame_bytes)
            raw10 = unpack_raw10(chunk, height, width)

            # shift left to 16 bit so OpenCV sees correct bit-depth
            raw16 = (raw10.astype(np.uint16) << 6)

            bgr16 = cv2.cvtColor(raw16, BAYER_MAP[bayer])

            # compress to 8 bit for video
            bgr8 = (bgr16 >> 8).astype(np.uint8)

            writer.write(bgr8)
            print(f"  frame {idx+1}/{n_frames}")

    writer.release()
    print(f"Saved {out_path}")



def compute_normalized_br(raw10, bayer="BGGR", black=64):
    # black level
    raw = raw10.astype(np.int32) - black
    raw = np.clip(raw, 0, 1023).astype(np.uint16)

    # scale to 16-bit for OpenCV demosaic
    raw16 = raw << 6

    # demosaic
    bgr16 = cv2.cvtColor(raw16, BAYER_MAP[bayer]).astype(np.float32)

    # OpenCV bruker BGR-rekkefølge
    b = bgr16[:, :, 0]
    r = bgr16[:, :, 2]

    eps = 1e-6
    #nbr = (b - r) / (b + r + eps)
    nbr = b / (r + eps)
    return nbr


def plot_hyta_histogram(nbr, bins=256, title="Histogram of normalized B/R ratio"):
    vals = nbr.flatten()

    plt.figure(figsize=(8, 4.5))
    plt.hist(vals, bins=bins, range=(-1, 1), density=False)
    plt.xlabel("Normalized B/R ratio  (B - R) / (B + R)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("std =", np.std(vals))



def show_nbr_image(nbr, title="Normalized B/R ratio image"):
    plt.figure(figsize=(6, 6))
    plt.imshow(nbr, cmap="gray", vmin=-1, vmax=1)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()








def raw10_to_images(
    raw_path: str,
    width: int,
    height: int,
    bayer: str,
    out_dir: str,
):
    frame_bytes = int(width * height * 1.25)
    file_size = os.path.getsize(raw_path)
    n_frames = file_size // frame_bytes

    if file_size % frame_bytes:
        raise ValueError(
            f"File size {file_size} is not an exact multiple of one RAW10 frame "
            f"({frame_bytes}). Check width/height."
        )

    os.makedirs(out_dir, exist_ok=True)

    print(f"{raw_path}: {n_frames} frames detected ({width}×{height} RAW10)")
    print(f"Saving images to: {out_dir}")

    with open(raw_path, "rb") as f:
        for idx in range(n_frames):
            chunk = f.read(frame_bytes)
            raw10 = unpack_raw10(chunk, height, width)

            nbr = compute_normalized_br(raw10, bayer=bayer, black=64)

            show_nbr_image(nbr, title="Normalized B/R ratio image")
            plot_hyta_histogram(nbr, title="HYTA-style histogram")
           
            # Save grayscale PGM from raw Bayer mosaic
            raw8 = (raw10 >> 2).astype(np.uint8)
            pgm_path = os.path.join(out_dir, f"frame_{idx:04d}.pgm")
            cv2.imwrite(pgm_path, raw8)
            """
            # Demosaic and save color PNG
            raw16 = raw10.astype(np.uint16) << 6
            bgr16 = cv2.cvtColor(raw16, BAYER_MAP[bayer])
            bgr8 = (bgr16 >> 8).astype(np.uint8)
            """
            raw = raw10.astype(np.int32)

            # 1) black level
            black = 64
            raw = raw - black
            raw = np.clip(raw, 0, 1023).astype(np.uint16)

            # 2) scale to 16-bit
            raw16 = raw << 6

            # 3) demosaic
            bgr16 = cv2.cvtColor(raw16, BAYER_MAP[bayer])


            # 4) white balance
            bgr = bgr16.astype(np.float32)
            bgr[..., 0] *= 1.5   # B
            bgr[..., 1] *= 1.0   # G
            bgr[..., 2] *= 1.5   # R
            bgr = np.clip(bgr, 0, 65535)

            # 5) gamma for display
            img = bgr / 65535.0
            img = np.clip(img, 0, 1)
            img = np.power(img, 1/2.2)

            bgr = bgr16.astype(np.float32)

            b_mean = np.mean(bgr[..., 0])
            g_mean = np.mean(bgr[..., 1])
            r_mean = np.mean(bgr[..., 2])

            bgr[..., 0] *= g_mean / max(b_mean, 1e-6)
            bgr[..., 2] *= g_mean / max(r_mean, 1e-6)

            bgr = np.clip(bgr, 0, 65535)
            scale = np.percentile(bgr, 99.5)
            img = np.clip(bgr / max(scale, 1.0), 0, 1)
            img = np.power(img, 1/2.2)
            bgr8 = (img * 255).astype(np.uint8)

            png_path = os.path.join(out_dir, f"frame_{idx:04d}.png")
            cv2.imwrite(png_path, bgr8)

            print(f"  saved {pgm_path}")
            print(f"  saved {png_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert multi-frame RAW10 dump to MP4 or numbered PNG/PGM images"
    )
    ap.add_argument("raw", help="input .raw file that holds N RAW10 frames")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument(
        "--bayer",
        choices=BAYER_MAP.keys(),
        default="BGGR",
        help="Bayer mosaic order (default BGGR)",
    )

    sub = ap.add_subparsers(dest="mode", required=True)

    video_parser = sub.add_parser("video", help="save output as MP4")
    video_parser.add_argument("--fps", type=int, default=30, help="output frame-rate")
    video_parser.add_argument("--out", default="output.mp4", help="output video path")

    image_parser = sub.add_parser("images", help="save all frames as numbered PNG and PGM files")
    image_parser.add_argument("--outdir", default="frames", help="output folder for files")

    args = ap.parse_args()

    if args.mode == "video":
        raw10_to_video(
            args.raw,
            args.width,
            args.height,
            args.bayer,
            args.fps,
            args.out,
        )
    elif args.mode == "images":
        raw10_to_images(
            args.raw,
            args.width,
            args.height,
            args.bayer,
            args.outdir,
        )