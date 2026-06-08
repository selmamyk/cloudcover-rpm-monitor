import numpy as np


def _normalized_br_ratio(rgb: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    HYTA feature:
      lam = B / R
      lamN = (lam - 1) / (lam + 1)

    Returns lamN roughly in [-1, 1).
    """
    rgb_f = rgb.astype(np.float32)
    r = rgb_f[..., 0]
    b = rgb_f[..., 2]
    lam = b / (r + eps)
    lamN = (lam - 1.0) / (lam + 1.0 + eps)
    return lamN.astype(np.float32)


def _to_levels(lamN, levels: int = 256) -> np.ndarray:
    """
    Map lamN in [-1,1] -> int levels [0..levels-1]
    """
    x = np.clip(lamN, -1.0, 1.0)
    x01 = (x + 1.0) / 2.0
    idx = np.floor(x01 * (levels - 1) + 1e-9).astype(np.int32)
    return idx


def _mce_threshold(level_img, levels = 256):
    """
    Minimum Cross Entropy thresholding on integer levels image.
    Returns t in [1..levels-2].
    """
    h = np.bincount(level_img.ravel(), minlength=levels).astype(np.float64)
    level_vals = np.arange(levels, dtype=np.float64)

    H = np.cumsum(h)
    IH = np.cumsum(h * level_vals)

    t_candidates = np.arange(1, levels - 1)
    H1 = H[t_candidates]
    H2 = H[-1] - H[t_candidates]
    valid = (H1 > 0) & (H2 > 0)

    t_candidates = t_candidates[valid]
    if t_candidates.size == 0:
        return levels // 2

    m1 = IH[t_candidates] / H1[valid]
    m2 = (IH[-1] - IH[t_candidates]) / H2[valid]

    ih = level_vals * h
    with np.errstate(divide="ignore", invalid="ignore"):
        ih_log_i = np.where(level_vals > 0, ih * np.log(level_vals), 0.0)

    IH_LOGI = np.cumsum(ih_log_i)

    S1 = IH_LOGI[t_candidates]
    S2 = IH_LOGI[-1] - IH_LOGI[t_candidates]

    A1 = IH[t_candidates]
    A2 = IH[-1] - IH[t_candidates]

    with np.errstate(divide="ignore", invalid="ignore"):
        D = (S1 - A1 * np.log(m1)) + (S2 - A2 * np.log(m2))

    t_star = int(t_candidates[np.argmin(D)])
    return t_star


def predict_mask(
    rgb,
    Tf = 0.250,
    sigma_thr = 0.03,
    levels = 256,
    eps = 1e-6,
    valid_mask: np.ndarray | None = None,
):
    """
    HYTA mask prediction.

    Args:
        rgb: uint8 RGB image (H,W,3)
        Tf: fixed threshold used when image is classified as unimodal
        sigma_thr: std-dev cutoff to decide unimodal vs bimodal
        levels: discretization levels for MCE
        eps: numeric stability
        valid_mask: optional boolean/uint8 mask where 1 marks pixels that
            should be included in the HYTA thresholding step

    Returns:
        mask uint8 (H,W) with {0,1} where 1=cloud
    """

    lamN = _normalized_br_ratio(rgb, eps=eps)
    if valid_mask is None:
        valid = np.ones(lamN.shape, dtype=bool)
    else:
        valid = valid_mask.astype(bool)
        if valid.shape != lamN.shape:
            raise ValueError(
                f"valid_mask must have shape {lamN.shape}, got {valid.shape}"
            )

    valid_lamn = lamN[valid]
    if valid_lamn.size == 0:
        raise ValueError("valid_mask excludes all pixels.")

    sigma = float(valid_lamn.std())

    if sigma < sigma_thr:
        thr = Tf
    else:
        level_img = _to_levels(valid_lamn, levels=levels)
        t_star = _mce_threshold(level_img, levels=levels)
        thr = (t_star / (levels - 1)) * 2.0 - 1.0  # back to [-1,1]

    # HYTA convention: cloud where lamN < threshold
    mask = ((lamN < thr) & valid).astype(np.uint8)
    return mask


def _threshold_from_lamn(
    lamN: np.ndarray,
    threshold_mask: np.ndarray,
    Tf: float,
    sigma_thr: float,
    levels: int,
) -> tuple[float, float]:
    valid_lamn = lamN[threshold_mask]
    if valid_lamn.size == 0:
        raise ValueError("threshold_mask excludes all pixels.")

    sigma = float(valid_lamn.std())
    if sigma < sigma_thr:
        return float(Tf), sigma

    level_img = _to_levels(valid_lamn, levels=levels)
    t_star = _mce_threshold(level_img, levels=levels)
    threshold = (t_star / (levels - 1)) * 2.0 - 1.0
    return float(threshold), sigma


def _center_mask(height: int, width: int, radius_scale: float) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = min(height, width) * float(radius_scale)
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2


def _bottom_boost_mask(
    lamN: np.ndarray,
    prediction_mask: np.ndarray,
    threshold: float,
    bottom_boost: float,
    bottom_boost_start: float,
) -> tuple[np.ndarray, float]:
    height, _width = lamN.shape
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    ramp = np.clip(
        (y - float(bottom_boost_start)) / max(1.0 - float(bottom_boost_start), 1e-6),
        0.0,
        1.0,
    )
    threshold_map = np.broadcast_to(threshold + float(bottom_boost) * ramp, lamN.shape)
    mean_threshold = float(np.mean(threshold_map[prediction_mask])) if np.any(prediction_mask) else float(threshold)
    return (lamN < threshold_map) & prediction_mask, mean_threshold


def _cap_mask_fraction(
    mask: np.ndarray,
    lamN: np.ndarray,
    prediction_mask: np.ndarray,
    max_fraction: float,
) -> np.ndarray:
    max_pixels = int(round(float(max_fraction) * int(prediction_mask.sum())))
    if max_pixels <= 0:
        return np.zeros_like(mask, dtype=bool)

    cloud_positions = np.flatnonzero(mask & prediction_mask)
    if cloud_positions.size <= max_pixels:
        return mask & prediction_mask

    flat_lamn = lamN.reshape(-1)
    keep_order = np.argpartition(flat_lamn[cloud_positions], max_pixels - 1)[:max_pixels]
    capped = np.zeros_like(mask, dtype=bool).reshape(-1)
    capped[cloud_positions[keep_order]] = True
    return capped.reshape(mask.shape)


def predict_mask_imx385_adapted(
    rgb,
    Tf: float = 0.05,
    sigma_thr: float = 0.01,
    levels: int = 256,
    eps: float = 1e-6,
    threshold_radius_scale: float = 0.55,
    prediction_radius_scale: float = 0.65,
    threshold_mask: np.ndarray | None = None,
    prediction_mask: np.ndarray | None = None,
    bottom_boost: float = 0.125,
    bottom_boost_start: float = 0.55,
    bottom_boost_gate: float = 0.35,
    low_okta_base_cf_max: float = 0.25,
    low_okta_cap_okta: int = 1,
):
    """
    IMX385-adapted HYTA variant with bottom boost and a low-okta guard.

    This keeps the original HYTA thresholding step, then:
      1. applies a gradually higher threshold in the lower part of the image
         when base HYTA finds enough cloud,
      2. caps the mask fraction for likely low-okta images.

    Args:
        rgb: uint8 RGB image (H,W,3)
        Tf, sigma_thr, levels, eps: same meaning as standard HYTA
        threshold_radius_scale: center-circle radius used for MCE thresholding
        prediction_radius_scale: center-circle radius used for cloud fraction
        threshold_mask: optional explicit bool mask for threshold estimation
        prediction_mask: optional explicit bool mask for output mask/fraction
        bottom_boost: max threshold increase at the bottom of the image
        bottom_boost_start: normalized y position where boost starts
        bottom_boost_gate: apply bottom boost only if base cloud fraction >= this
        low_okta_base_cf_max: cap only if base cloud fraction <= this
        low_okta_cap_okta: max okta allowed by the cap, expressed in oktas

    Returns:
        mask uint8 (H,W) with {0,1} where 1=cloud
    """
    lamN = _normalized_br_ratio(rgb, eps=eps)
    height, width = lamN.shape

    if prediction_mask is None:
        pred_valid = _center_mask(height, width, prediction_radius_scale)
    else:
        pred_valid = prediction_mask.astype(bool)
        if pred_valid.shape != lamN.shape:
            raise ValueError(f"prediction_mask must have shape {lamN.shape}, got {pred_valid.shape}")

    if threshold_mask is None:
        thr_valid = _center_mask(height, width, threshold_radius_scale)
    else:
        thr_valid = threshold_mask.astype(bool)
        if thr_valid.shape != lamN.shape:
            raise ValueError(f"threshold_mask must have shape {lamN.shape}, got {thr_valid.shape}")

    threshold, _sigma = _threshold_from_lamn(lamN, thr_valid, Tf, sigma_thr, levels)
    base_mask = (lamN < threshold) & pred_valid
    base_cloud_fraction = float(base_mask[pred_valid].mean()) if np.any(pred_valid) else 0.0

    if base_cloud_fraction >= bottom_boost_gate:
        mask, _boost_threshold = _bottom_boost_mask(
            lamN,
            pred_valid,
            threshold,
            bottom_boost,
            bottom_boost_start,
        )
    else:
        mask = base_mask

    if base_cloud_fraction <= low_okta_base_cf_max:
        mask = _cap_mask_fraction(mask, lamN, pred_valid, float(low_okta_cap_okta) / 8.0)

    return mask.astype(np.uint8)

