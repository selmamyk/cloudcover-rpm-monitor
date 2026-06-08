from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


_BAYER_TO_RGB = {
    "BGGR": cv2.COLOR_BayerBG2RGB,
    "RGGB": cv2.COLOR_BayerRG2RGB,
    "GRBG": cv2.COLOR_BayerGR2RGB,
    "GBRG": cv2.COLOR_BayerGB2RGB,
}


def unpack_raw10_frame(chunk: bytes, height: int, width: int) -> np.ndarray:
    """
    Unpack one packed RAW10 frame into a uint16 array of shape (H, W).

    RAW10 stores 4 pixels in 5 bytes.
    """
    expected_bytes = width * height * 5 // 4
    if len(chunk) != expected_bytes:
        raise ValueError(
            f"Expected {expected_bytes} bytes for one RAW10 frame, got {len(chunk)}."
        )

    packed = np.frombuffer(chunk, dtype=np.uint8).reshape(-1, 5)

    p0 = (packed[:, 0].astype(np.uint16) << 2) | ((packed[:, 4] >> 0) & 0x03)
    p1 = (packed[:, 1].astype(np.uint16) << 2) | ((packed[:, 4] >> 2) & 0x03)
    p2 = (packed[:, 2].astype(np.uint16) << 2) | ((packed[:, 4] >> 4) & 0x03)
    p3 = (packed[:, 3].astype(np.uint16) << 2) | ((packed[:, 4] >> 6) & 0x03)

    unpacked = np.empty(packed.shape[0] * 4, dtype=np.uint16)
    unpacked[0::4] = p0
    unpacked[1::4] = p1
    unpacked[2::4] = p2
    unpacked[3::4] = p3
    return unpacked.reshape(height, width)


def load_raw10_frame(
    path: str | Path,
    width: int,
    height: int,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Load one frame from a packed RAW10 file.

    Supports files that contain either one frame or multiple concatenated frames.
    """
    path = Path(path)
    frame_bytes = width * height * 5 // 4
    file_size = path.stat().st_size

    if file_size % frame_bytes != 0:
        raise ValueError(
            f"File size {file_size} is not a multiple of one RAW10 frame "
            f"({frame_bytes} bytes). Check width/height."
        )

    n_frames = file_size // frame_bytes
    if frame_idx < 0 or frame_idx >= n_frames:
        raise ValueError(f"frame_idx must be in [0, {n_frames - 1}]")

    with path.open("rb") as handle:
        handle.seek(frame_idx * frame_bytes)
        chunk = handle.read(frame_bytes)

    return unpack_raw10_frame(chunk, height=height, width=width)


def subtract_black_level(
    raw10: np.ndarray,
    black_level: int = 64,
    saturation_level: int = 1023,
) -> np.ndarray:
    """
    Subtract sensor black level and clamp to the valid RAW10 range.
    """
    corrected = raw10.astype(np.int32) - int(black_level)
    corrected = np.clip(corrected, 0, int(saturation_level) - int(black_level))
    return corrected.astype(np.uint16)


def gray_world_white_balance(rgb16: np.ndarray) -> np.ndarray:
    """
    Apply a simple gray-world white balance in linear space.
    """
    rgb = rgb16.astype(np.float32)
    means = rgb.mean(axis=(0, 1))
    g_mean = max(float(means[1]), 1e-6)

    gains = np.array(
        [
            g_mean / max(float(means[0]), 1e-6),
            1.0,
            g_mean / max(float(means[2]), 1e-6),
        ],
        dtype=np.float32,
    )
    balanced = rgb * gains
    return np.clip(balanced, 0, 65535).astype(np.uint16)


def demosaic_raw10_to_rgb(
    raw10: np.ndarray,
    bayer: str = "BGGR",
    black_level: int = 64,
    white_balance: bool = True,
) -> np.ndarray:
    """
    Convert a RAW10 Bayer frame to display-ready RGB uint8.

    The returned image is suitable as input to the HYTA implementation.
    """
    bayer = bayer.upper()
    if bayer not in _BAYER_TO_RGB:
        raise ValueError(f"Unsupported Bayer pattern: {bayer}")

    corrected = subtract_black_level(raw10, black_level=black_level)
    rgb16 = cv2.cvtColor(corrected << 6, _BAYER_TO_RGB[bayer])

    if white_balance:
        rgb16 = gray_world_white_balance(rgb16)

    # Contrast stretch before gamma compression to keep cloud structure visible.
    scale = float(np.percentile(rgb16, 99.5))
    scale = max(scale, 1.0)
    rgb = np.clip(rgb16.astype(np.float32) / scale, 0.0, 1.0)
    rgb = np.power(rgb, 1.0 / 2.2)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

