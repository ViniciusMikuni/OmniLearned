"""Binary classifier metrics with attainable empirical ROC working points."""

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def binary_metrics(labels, prediction, efficiencies=(0.7, 0.3)):
    """Class 1 is signal; use the first threshold attaining at least each target.

    Tied scores can overshoot the requested signal efficiency. Return the actual
    value and threshold. Infinite rejection means zero observed false positives.
    """
    labels = np.asarray(labels).reshape(-1)
    prediction = np.asarray(prediction)
    if prediction.shape != (len(labels), 2) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Expected both labels 0/1 and prediction shape (N, 2)")
    if not np.isfinite(prediction).all():
        raise ValueError("Predictions must be finite")
    if ((prediction < 0) | (prediction > 1)).any() or not np.allclose(
        prediction.sum(1), 1, atol=1e-5
    ):
        raise ValueError("Predictions must be two-class probabilities")
    score = prediction[:, 1]
    fpr, tpr, thresholds = roc_curve(labels, score, drop_intermediate=False)
    points = []
    for target in efficiencies:
        if not 0 < target <= 1:
            raise ValueError("Signal efficiencies must be in (0, 1]")
        i = np.flatnonzero(tpr >= target)[0]
        points.append(
            {
                "target_signal_efficiency": float(target),
                "signal_efficiency": float(tpr[i]),
                "background_efficiency": float(fpr[i]),
                "background_rejection": float(1 / fpr[i]) if fpr[i] else float("inf"),
                "threshold": float(thresholds[i]),
            }
        )
    return {
        "accuracy": float(accuracy_score(labels, prediction.argmax(1))),
        "auc": float(roc_auc_score(labels, score)),
        "working_points": points,
    }
