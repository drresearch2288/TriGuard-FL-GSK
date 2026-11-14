import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score, roc_auc_score


def balanced_accuracy(y, pred, num_classes=None) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    recalls = [np.mean(pred[y == c] == c) for c in np.unique(y)]
    return float(np.mean(recalls))


def macro_auc(y, logits) -> float:
    p = softmax(logits, 1)
    present = np.unique(y)
    if p.shape[1] == 2:
        return float(roc_auc_score(y, p[:, 1])) if len(present) == 2 else float("nan")
    aucs = [roc_auc_score(y == c, p[:, c]) for c in present if 0 < (y == c).sum() < len(y)]
    return float(np.mean(aucs)) if aucs else float("nan")


def macro_f1(y, pred) -> float:
    return float(f1_score(y, pred, average="macro"))


def ece(y, probs, bins=15) -> float:
    conf = probs.max(1)
    acc = probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def brier(y, probs) -> float:
    oh = np.eye(probs.shape[1])[y]
    return float(((probs - oh) ** 2).sum(1).mean())


def reliability(y, probs, bins=15):
    conf = probs.max(1)
    acc = probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append((float((lo + hi) / 2), float(acc[m].mean()) if m.any() else np.nan, int(m.sum())))
    return rows
