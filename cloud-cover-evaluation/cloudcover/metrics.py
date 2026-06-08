from __future__ import annotations
import numpy as np


def metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-9) -> dict:
    """
    Compute metrics for binary masks.

    Args:
        pred: predicted mask, values {0,1}, shape (H,W)
        gt: ground truth mask, values {0,1}, shape (H,W)
        eps: numerical stability

    Returns:
        dict with:
          tp, tn, fp, fn,
          accuracy, precision, recall, f1, dice, iou,
          pred_cloud_fraction, gt_cloud_fraction, cloud_abs_error
    """

    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    tp = int(np.logical_and(pred_b, gt_b).sum())
    tn = int(np.logical_and(~pred_b, ~gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())

    total = tp + tn + fp + fn + eps

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)

    pred_cloud_fraction = float(pred_b.mean())
    gt_cloud_fraction = float(gt_b.mean())
    cloud_abs_error = abs(pred_cloud_fraction - gt_cloud_fraction)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dice": float(dice),
        "iou": float(iou),
        "pred_cloud_fraction": pred_cloud_fraction,
        "gt_cloud_fraction": gt_cloud_fraction,
        "cloud_abs_error": float(cloud_abs_error),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

def error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Create an RGB error map for visualization:
      - Black  : True Negative (correct sky)
      - White  : True Positive (correct cloud)
      - Red    : False Positive (pred cloud, GT sky)
      - Blue   : False Negative (GT cloud, pred sky)

    Returns:
        uint8 RGB image (H,W,3)
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    h, w = pred_b.shape
    err = np.zeros((h, w, 3), dtype=np.uint8)

    # TP -> white
    err[np.logical_and(pred_b, gt_b)] = [255, 255, 255]
    # FP -> red
    err[np.logical_and(pred_b, ~gt_b)] = [255, 0, 0]
    # FN -> blue
    err[np.logical_and(~pred_b, gt_b)] = [0, 0, 255]
    # TN stays black
    return err

