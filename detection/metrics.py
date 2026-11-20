import numpy as np


def binary_metrics(labels, predicted, scores):
    """Evaluate 0=real and 1=AI. Average ranks for tied AUROC scores."""
    labels = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != predicted.shape or labels.shape != scores.shape:
        raise ValueError("Labels, classes and scores must be aligned one-dimensional arrays")
    if not np.isin(labels, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Labels and predicted classes must be binary")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite validation scores")

    matrix = np.bincount(2 * labels + predicted, minlength=4).reshape(2, 2)
    negatives, positives = matrix.sum(axis=1)
    if not negatives or not positives:
        raise ValueError("Both classes are required for AUROC")

    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1, scores.size]
    starts, ends = boundaries[:-1], boundaries[1:]
    ranks = np.repeat((starts + 1 + ends) / 2, ends - starts)
    positive_rank_sum = ranks[labels[order] == 1].sum()
    return {
        "n": int(labels.size),
        "accuracy": float(matrix.trace() / labels.size),
        "auroc": float((positive_rank_sum - positives * (positives + 1) / 2)
                       / (positives * negatives)),
        "recall_real": float(matrix[0, 0] / negatives),
        "recall_ai": float(matrix[1, 1] / positives),
        "confusion_matrix": matrix.tolist(),
    }
