import numpy as np

def predict_mask(rgb, thr=0.7, eps=1e-6):
    r = rgb[:, :, 0].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    ratio = r / (b + eps)
    return (ratio > thr).astype(np.uint8)
