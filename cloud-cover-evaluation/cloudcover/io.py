from pathlib import Path
import cv2
import numpy as np

def load_rgb(path):
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def load_gt_mask(path):
    gt = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f"Could not read GT mask: {path}")
    return (gt > 0).astype(np.uint8)
