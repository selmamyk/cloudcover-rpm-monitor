import cv2
import numpy as np

def predict_mask(rgb, s_thr=0.25, v_min=0.20, v_max=1.0, blur=0):
    if rgb.dtype != np.uint8:
        raise ValueError("rgb must be uint8 (0..255)")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    if blur and blur > 1:
        if blur % 2 == 0:
            raise ValueError("blur must be odd or 0")
        s = cv2.GaussianBlur(s, (blur, blur), 0)
        v = cv2.GaussianBlur(v, (blur, blur), 0)

    cloud = (s < s_thr) & (v >= v_min) & (v <= v_max)
    return cloud.astype(np.uint8)
