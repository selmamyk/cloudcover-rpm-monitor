from __future__ import annotations

import cv2
import numpy as np


def normalized_br_ratio(rgb: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    r = rgb_f[..., 0]
    b = rgb_f[..., 2]
    lam = b / (r + eps)
    lamn = (lam - 1.0) / (lam + 1.0 + eps)
    return lamn.astype(np.float32)


def to_levels(lamn: np.ndarray, levels: int = 256) -> np.ndarray:
    x = np.clip(lamn, -1.0, 1.0)
    x01 = (x + 1.0) / 2.0
    return np.floor(x01 * (levels - 1) + 1e-9).astype(np.int32)


def mce_threshold(level_img: np.ndarray, levels: int = 256) -> int:
    hist = np.bincount(level_img.ravel(), minlength=levels).astype(np.float64)
    level_vals = np.arange(levels, dtype=np.float64)
    cum_hist = np.cumsum(hist)
    cum_intensity = np.cumsum(hist * level_vals)

    candidates = np.arange(1, levels - 1)
    left_hist = cum_hist[candidates]
    right_hist = cum_hist[-1] - cum_hist[candidates]
    valid = (left_hist > 0) & (right_hist > 0)
    candidates = candidates[valid]
    if candidates.size == 0:
        return levels // 2

    m1 = cum_intensity[candidates] / left_hist[valid]
    m2 = (cum_intensity[-1] - cum_intensity[candidates]) / right_hist[valid]

    ih = level_vals * hist
    with np.errstate(divide="ignore", invalid="ignore"):
        ih_log_i = np.where(level_vals > 0, ih * np.log(level_vals), 0.0)

    cum_log = np.cumsum(ih_log_i)
    s1 = cum_log[candidates]
    s2 = cum_log[-1] - cum_log[candidates]
    a1 = cum_intensity[candidates]
    a2 = cum_intensity[-1] - cum_intensity[candidates]

    with np.errstate(divide="ignore", invalid="ignore"):
        distance = (s1 - a1 * np.log(m1)) + (s2 - a2 * np.log(m2))

    return int(candidates[np.argmin(distance)])


def predict_hyta_mask(
    rgb: np.ndarray,
    valid_mask: np.ndarray | None = None,
    threshold_mask: np.ndarray | None = None,
    tf: float = 0.15,
    sigma_thr: float = 0.01,
    levels: int = 256,
) -> tuple[np.ndarray, float, float]:
    lamn = normalized_br_ratio(rgb)
    if valid_mask is None:
        valid = np.ones(lamn.shape, dtype=bool)
    else:
        valid = valid_mask.astype(bool)

    if threshold_mask is None:
        threshold_valid = valid
    else:
        threshold_valid = threshold_mask.astype(bool) & valid

    valid_lamn = lamn[threshold_valid]
    if valid_lamn.size == 0:
        raise ValueError("HYTA threshold mask excludes all valid pixels.")

    sigma = float(valid_lamn.std())
    if sigma < sigma_thr:
        thr = tf
    else:
        level_img = to_levels(valid_lamn, levels=levels)
        t_star = mce_threshold(level_img, levels=levels)
        thr = (t_star / (levels - 1)) * 2.0 - 1.0

    mask = ((lamn < thr) & valid).astype(np.uint8)
    return mask, float(thr), sigma


def bottom_boost_mask(
    lamn: np.ndarray,
    prediction_mask: np.ndarray,
    threshold: float,
    bottom_boost: float,
    bottom_boost_start: float,
) -> tuple[np.ndarray, float]:
    height, _width = lamn.shape
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    ramp = np.clip(
        (y - float(bottom_boost_start)) / max(1.0 - float(bottom_boost_start), 1e-6),
        0.0,
        1.0,
    )
    threshold_map = np.broadcast_to(threshold + float(bottom_boost) * ramp, lamn.shape)
    mean_threshold = (
        float(np.mean(threshold_map[prediction_mask]))
        if np.any(prediction_mask)
        else float(threshold)
    )
    return (lamn < threshold_map) & prediction_mask, mean_threshold


def cap_mask_fraction(
    mask: np.ndarray,
    lamn: np.ndarray,
    prediction_mask: np.ndarray,
    max_fraction: float,
) -> np.ndarray:
    max_pixels = int(round(float(max_fraction) * int(prediction_mask.sum())))
    if max_pixels <= 0:
        return np.zeros_like(mask, dtype=bool)

    cloud_positions = np.flatnonzero(mask & prediction_mask)
    if cloud_positions.size <= max_pixels:
        return mask & prediction_mask

    flat_lamn = lamn.reshape(-1)
    keep_order = np.argpartition(flat_lamn[cloud_positions], max_pixels - 1)[:max_pixels]
    capped = np.zeros_like(mask, dtype=bool).reshape(-1)
    capped[cloud_positions[keep_order]] = True
    return capped.reshape(mask.shape)


def predict_hyta_mask_imx385_adapted(
    rgb: np.ndarray,
    valid_mask: np.ndarray | None = None,
    threshold_mask: np.ndarray | None = None,
    tf: float = 0.05,
    sigma_thr: float = 0.01,
    levels: int = 256,
    bottom_boost: float = 0.125,
    bottom_boost_start: float = 0.55,
    bottom_boost_gate: float = 0.35,
    low_okta_base_cf_max: float = 0.25,
    low_okta_cap_okta: int = 1,
) -> tuple[np.ndarray, float, float]:
    lamn = normalized_br_ratio(rgb)
    if valid_mask is None:
        prediction_mask = np.ones(lamn.shape, dtype=bool)
    else:
        prediction_mask = valid_mask.astype(bool)

    if threshold_mask is None:
        threshold_valid = prediction_mask
    else:
        threshold_valid = threshold_mask.astype(bool) & prediction_mask

    valid_lamn = lamn[threshold_valid]
    if valid_lamn.size == 0:
        raise ValueError("HYTA threshold mask excludes all valid pixels.")

    sigma = float(valid_lamn.std())
    if sigma < sigma_thr:
        threshold = float(tf)
    else:
        level_img = to_levels(valid_lamn, levels=levels)
        t_star = mce_threshold(level_img, levels=levels)
        threshold = float((t_star / (levels - 1)) * 2.0 - 1.0)

    base_mask = (lamn < threshold) & prediction_mask
    base_cloud_fraction = (
        float(base_mask[prediction_mask].mean()) if np.any(prediction_mask) else 0.0
    )

    if base_cloud_fraction >= bottom_boost_gate:
        mask, threshold = bottom_boost_mask(
            lamn,
            prediction_mask,
            threshold,
            bottom_boost,
            bottom_boost_start,
        )
    else:
        mask = base_mask

    if base_cloud_fraction <= low_okta_base_cf_max:
        mask = cap_mask_fraction(
            mask,
            lamn,
            prediction_mask,
            float(low_okta_cap_okta) / 8.0,
        )

    return mask.astype(np.uint8), float(threshold), sigma


def make_hyta_valid_mask(height: int, width: int, radius_scale: float = 0.60) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = min(height, width) * radius_scale
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2).astype(np.uint8)


def overlay_cloud_mask(
    bgr: np.ndarray,
    cloud_mask: np.ndarray,
    valid_mask: np.ndarray | None = None,
    alpha: float = 0.35,
) -> np.ndarray:
    overlay = bgr.astype(np.float32).copy()
    red = np.zeros_like(overlay)
    red[..., 2] = 255.0
    cloud_pixels = cloud_mask.astype(bool)
    overlay[cloud_pixels] = (
        (1.0 - alpha) * overlay[cloud_pixels] + alpha * red[cloud_pixels]
    )

    if valid_mask is not None:
        contours, _ = cv2.findContours(
            valid_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

    return np.clip(overlay, 0, 255).astype(np.uint8)
