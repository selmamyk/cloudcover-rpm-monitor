import numpy as np


def predict_mask(rgb, thr = -0.25, eps = 1e-6):
    """
    Normalized Difference Red-Blue (NDRB):
        score = (R - B) / (R + B + eps)

    Args:
        rgb: uint8 RGB image (H, W, 3)
        thr: threshold on score (typically in [-1, 1])
        eps: avoid division by zero

    Returns:
        mask uint8 (H, W) with {0,1} where 1=cloud
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")

    r = rgb[:, :, 0].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    score = (r - b) / (r + b + eps)

    # Convention: cloud where score > thr (tune thr on your validation set)
    mask = (score > thr).astype(np.uint8)
    return mask
