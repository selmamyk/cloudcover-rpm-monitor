from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hyta import (
    make_hyta_valid_mask,
    overlay_cloud_mask,
    predict_hyta_mask,
    predict_hyta_mask_imx385_adapted,
)


BAYER_TO_BGR = {
    "BGGR": cv2.COLOR_BayerBG2BGR,
    "RGGB": cv2.COLOR_BayerRG2BGR,
    "GRBG": cv2.COLOR_BayerGR2BGR,
    "GBRG": cv2.COLOR_BayerGB2BGR,
}


def cloud_fraction_to_okta(cloud_fraction: float) -> int:
    return int(np.clip(round(cloud_fraction * 8.0), 0, 8))


def normalize_hyta_method(method: str) -> str:
    if method == "low_okta_guard":
        return "imx385_adapted"
    if method in {"hyta", "imx385_adapted"}:
        return method
    raise ValueError(f"Unknown HYTA method: {method}")


def hyta_method_label(method: str) -> str:
    normalized = normalize_hyta_method(method)
    if normalized == "imx385_adapted":
        return "IMX385-adapted"
    return "HYTA"


def unpack_raw10(raw_path: Path, width: int, height: int) -> np.ndarray:
    data = raw_path.read_bytes()
    expected = width * height * 5 // 4
    if len(data) != expected:
        raise ValueError(f"Unexpected RAW size {len(data)} bytes, expected {expected}")

    packed = np.frombuffer(data, dtype=np.uint8).reshape(-1, 5)
    p0 = (packed[:, 0].astype(np.uint16) << 2) | ((packed[:, 4] >> 0) & 0x03)
    p1 = (packed[:, 1].astype(np.uint16) << 2) | ((packed[:, 4] >> 2) & 0x03)
    p2 = (packed[:, 2].astype(np.uint16) << 2) | ((packed[:, 4] >> 4) & 0x03)
    p3 = (packed[:, 3].astype(np.uint16) << 2) | ((packed[:, 4] >> 6) & 0x03)

    out = np.empty(packed.shape[0] * 4, dtype=np.uint16)
    out[0::4] = p0
    out[1::4] = p1
    out[2::4] = p2
    out[3::4] = p3
    return out.reshape((height, width))


def build_center_mask(height: int, width: int, radius_scale: float = 0.35) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = min(height, width) * radius_scale
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2


def measure_frame_quality(raw10: np.ndarray, black_level: int = 60) -> dict:
    corrected = np.clip(raw10.astype(np.int32) - black_level, 0, 1023).astype(np.uint16)
    valid = corrected[build_center_mask(*corrected.shape)]

    if valid.size == 0:
        valid = corrected.reshape(-1)

    mean_value = float(valid.mean())
    median_value = float(np.median(valid))
    p95 = float(np.percentile(valid, 95))
    p99 = float(np.percentile(valid, 99))
    dark_fraction = float((valid < 80).mean())
    clipped_fraction = float((valid > 1000).mean())
    return {
        "mean": mean_value,
        "median": median_value,
        "p95": p95,
        "p99": p99,
        "dark_fraction": dark_fraction,
        "clipped_fraction": clipped_fraction,
    }


def raw10_to_preview_bgr(
    raw10: np.ndarray,
    bayer_pattern: str = "BGGR",
    black_level: int = 60,
) -> np.ndarray:
    corrected = np.clip(raw10.astype(np.int32) - black_level, 0, 1023).astype(np.uint16)
    bayer_code = BAYER_TO_BGR[bayer_pattern.upper()]
    bgr16 = cv2.cvtColor(corrected << 6, bayer_code).astype(np.float32)

    means = bgr16.mean(axis=(0, 1))
    g_mean = max(float(means[1]), 1e-6)
    if means[0] > 0:
        bgr16[..., 0] *= g_mean / float(means[0])
    if means[2] > 0:
        bgr16[..., 2] *= g_mean / float(means[2])

    scale = max(float(np.percentile(bgr16, 99.5)), 1.0)
    img = np.clip(bgr16 / scale, 0.0, 1.0)
    img = np.power(img, 1.0 / 2.2)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def save_bgr_image(image: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def render_capture_preview(
    raw_path: Path,
    out_path: Path,
    width: int,
    height: int,
    bayer_pattern: str,
    caption_lines: list[str] | None = None,
) -> None:
    raw10 = unpack_raw10(raw_path, width, height)
    preview = raw10_to_preview_bgr(raw10, bayer_pattern=bayer_pattern)

    if caption_lines:
        y = 30
        for line in caption_lines:
            cv2.putText(
                preview,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 32

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), preview)


def save_capture_debug_outputs(
    raw_path: Path,
    sample_dir: Path,
    base_dir: Path,
    width: int,
    height: int,
    bayer_pattern: str,
    controls: dict,
    ir_cut_status: str | None,
    enable_hyta: bool,
    hyta_tf: float,
    hyta_sigma_thr: float,
    threshold_radius_scale: float,
    prediction_radius_scale: float,
    hyta_method: str,
    bottom_boost: float,
    bottom_boost_start: float,
    bottom_boost_gate: float,
    low_okta_base_cf_max: float,
    low_okta_cap_okta: int,
    save_image_preview: bool,
    save_hyta_debug_image: bool,
) -> dict:
    raw10 = unpack_raw10(raw_path, width, height)
    quality = measure_frame_quality(raw10)

    if save_image_preview:
        capture_caption = [
            f"captured sample={sample_dir.name}",
            f"exp={controls.get('exposure', 0)} again={controls.get('analogue_gain', 0)} gain={controls.get('gain', 0)}",
            f"ir_cut={ir_cut_status}",
        ]
        render_capture_preview(
            raw_path=raw_path,
            out_path=sample_dir / "captured_preview.png",
            width=width,
            height=height,
            bayer_pattern=bayer_pattern,
            caption_lines=capture_caption,
        )
        render_capture_preview(
            raw_path=raw_path,
            out_path=base_dir / "latest_capture.jpg",
            width=width,
            height=height,
            bayer_pattern=bayer_pattern,
            caption_lines=capture_caption,
        )

    hyta_result = None
    if enable_hyta:
        preview = raw10_to_preview_bgr(raw10, bayer_pattern=bayer_pattern)
        hyta_result = predict_hyta_for_bgr(
            preview=preview,
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

    if save_hyta_debug_image:
        hyta_result = save_debug_preview(
            raw10=raw10,
            output_path=sample_dir / "captured_hyta_debug.png",
            controls=controls,
            quality=quality,
            bayer_pattern=bayer_pattern,
            ir_cut_status=ir_cut_status,
            latest_path=base_dir / "latest_debug.jpg",
            enable_hyta=enable_hyta,
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
    if hyta_result is not None:
        quality["hyta"] = hyta_result
    return quality


def predict_hyta_for_bgr(
    preview: np.ndarray,
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
) -> dict:
    hyta_method = normalize_hyta_method(hyta_method)
    preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    prediction_mask = make_hyta_valid_mask(
        preview.shape[0],
        preview.shape[1],
        radius_scale=prediction_radius_scale,
    )
    threshold_mask = make_hyta_valid_mask(
        preview.shape[0],
        preview.shape[1],
        radius_scale=threshold_radius_scale,
    )

    if hyta_method == "hyta":
        hyta_mask, hyta_thr, hyta_sigma = predict_hyta_mask(
            preview_rgb,
            valid_mask=prediction_mask,
            threshold_mask=threshold_mask,
            tf=hyta_tf,
            sigma_thr=hyta_sigma_thr,
        )
    elif hyta_method == "imx385_adapted":
        hyta_mask, hyta_thr, hyta_sigma = predict_hyta_mask_imx385_adapted(
            preview_rgb,
            valid_mask=prediction_mask,
            threshold_mask=threshold_mask,
            tf=hyta_tf,
            sigma_thr=hyta_sigma_thr,
            bottom_boost=bottom_boost,
            bottom_boost_start=bottom_boost_start,
            bottom_boost_gate=bottom_boost_gate,
            low_okta_base_cf_max=low_okta_base_cf_max,
            low_okta_cap_okta=low_okta_cap_okta,
        )
    else:
        raise ValueError(f"Unknown HYTA method: {hyta_method}")
    cloud_fraction = (
        float(hyta_mask[prediction_mask.astype(bool)].mean())
        if prediction_mask.any()
        else 0.0
    )
    okta = cloud_fraction_to_okta(cloud_fraction)
    return {
        "cloud_fraction": cloud_fraction,
        "cloud_percent": cloud_fraction * 100.0,
        "okta": okta,
        "okta_text": f"{okta}/8",
        "threshold": hyta_thr,
        "sigma": hyta_sigma,
        "method": hyta_method,
        "bottom_boost": bottom_boost,
        "bottom_boost_start": bottom_boost_start,
        "bottom_boost_gate": bottom_boost_gate,
        "low_okta_base_cf_max": low_okta_base_cf_max,
        "low_okta_cap_okta": low_okta_cap_okta,
        "threshold_radius_scale": threshold_radius_scale,
        "prediction_radius_scale": prediction_radius_scale,
    }


def save_debug_preview(
    raw10: np.ndarray,
    output_path: Path,
    controls: dict,
    quality: dict,
    bayer_pattern: str,
    ir_cut_status: str | None = None,
    latest_path: Path | None = None,
    enable_hyta: bool = False,
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
) -> dict | None:
    hyta_method = normalize_hyta_method(hyta_method)
    preview = raw10_to_preview_bgr(raw10, bayer_pattern=bayer_pattern)
    hyta_result = None
    cloud_fraction_text = None
    if enable_hyta:
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        prediction_mask = make_hyta_valid_mask(
            preview.shape[0],
            preview.shape[1],
            radius_scale=prediction_radius_scale,
        )
        threshold_mask = make_hyta_valid_mask(
            preview.shape[0],
            preview.shape[1],
            radius_scale=threshold_radius_scale,
        )
        if hyta_method == "hyta":
            hyta_mask, hyta_thr, hyta_sigma = predict_hyta_mask(
                preview_rgb,
                valid_mask=prediction_mask,
                threshold_mask=threshold_mask,
                tf=hyta_tf,
                sigma_thr=hyta_sigma_thr,
            )
        elif hyta_method == "imx385_adapted":
            hyta_mask, hyta_thr, hyta_sigma = predict_hyta_mask_imx385_adapted(
                preview_rgb,
                valid_mask=prediction_mask,
                threshold_mask=threshold_mask,
                tf=hyta_tf,
                sigma_thr=hyta_sigma_thr,
                bottom_boost=bottom_boost,
                bottom_boost_start=bottom_boost_start,
                bottom_boost_gate=bottom_boost_gate,
                low_okta_base_cf_max=low_okta_base_cf_max,
                low_okta_cap_okta=low_okta_cap_okta,
            )
        else:
            raise ValueError(f"Unknown HYTA method: {hyta_method}")
        overlay = overlay_cloud_mask(
            preview,
            hyta_mask,
            valid_mask=prediction_mask,
        )
        mask_panel = np.dstack([hyta_mask * 255] * 3).astype(np.uint8)
        threshold_contours, _ = cv2.findContours(
            threshold_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(mask_panel, threshold_contours, -1, (255, 0, 255), 2)
        preview = np.hstack([preview, overlay, mask_panel])
        cloud_fraction = (
            float(hyta_mask[prediction_mask.astype(bool)].mean())
            if prediction_mask.any()
            else 0.0
        )
        okta = cloud_fraction_to_okta(cloud_fraction)
        cloud_fraction_text = (
            f"{hyta_method_label(hyta_method)} cloud={cloud_fraction:.1%} okta={okta}/8 thr={hyta_thr:.3f} sigma={hyta_sigma:.4f}"
        )
        hyta_result = {
            "cloud_fraction": cloud_fraction,
            "cloud_percent": cloud_fraction * 100.0,
            "okta": okta,
            "okta_text": f"{okta}/8",
            "threshold": hyta_thr,
            "sigma": hyta_sigma,
            "method": hyta_method,
            "bottom_boost": bottom_boost,
            "bottom_boost_start": bottom_boost_start,
            "bottom_boost_gate": bottom_boost_gate,
            "low_okta_base_cf_max": low_okta_base_cf_max,
            "low_okta_cap_okta": low_okta_cap_okta,
            "threshold_radius_scale": threshold_radius_scale,
            "prediction_radius_scale": prediction_radius_scale,
        }

    text_lines = [
        f"exp={controls['exposure']} again={controls['analogue_gain']} gain={controls['gain']}",
        f"mean={quality['mean']:.1f} p99={quality['p99']:.1f} dark={quality['dark_fraction']:.3f}",
    ]
    if ir_cut_status is not None:
        text_lines.append(f"ir_cut={ir_cut_status}")
    if cloud_fraction_text is not None:
        text_lines.append(cloud_fraction_text)

    y = 30
    for line in text_lines:
        cv2.putText(
            preview,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 32

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), preview)
    if latest_path is not None:
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(latest_path), preview)
    return hyta_result
