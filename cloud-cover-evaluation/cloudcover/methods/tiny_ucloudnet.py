from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf

_MODEL = None

def _get_model(model_path):
    global _MODEL
    if _MODEL is None:
        _MODEL = tf.keras.models.load_model(model_path, compile=False)
    return _MODEL


def predict_mask(rgb, model_path="tiny_ucloudnet_pc.h5", img_size=304, thr=0.5):
    """
    rgb: input image as HxWx3 numpy array
    returns: binary mask with same height/width as input
    """
    model = _get_model(model_path)

    h, w = rgb.shape[:2]

    # preprocess
    img = cv2.resize(rgb, (img_size, img_size)).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # (1, img_size, img_size, 3)

    # predict
    pred = model.predict(img, verbose=0)[0]   # (img_size, img_size, 1)
    pred = pred.squeeze()

    # threshold
    mask = (pred > thr).astype(np.uint8)

    # resize back to original size
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask
