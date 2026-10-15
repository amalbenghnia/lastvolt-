"""
Isotonic calibration for the EnergyMind V2 Bayesian diagnoser.

The Naive Bayes model produces overconfident or underdispersed posteriors.
Isotonic calibration corrects this post-hoc without retraining the base model.

Method: one-vs-rest isotonic regression per class, trained on a held-out
calibration split that is strictly separate from training and test sets.

References:
  Zadrozny & Elkan (2002) — Transforming classifier scores into accurate
    multiclass probability estimates.
  Niculescu-Mizil & Caruana (2005) — Predicting good probabilities with
    supervised learning.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import label_binarize


class IsotonicCalibrator:
    """
    One-vs-rest isotonic calibration for the Bayesian diagnoser.

    One isotonic regression is fit per class: P(H_i | score_i) is
    guaranteed to be monotone non-decreasing in score_i.

    Usage:
        cal = IsotonicCalibrator(classes)
        cal.fit(posteriors_cal, labels_cal)
        p_cal = cal.transform(posteriors_test)
    """

    def __init__(self, classes: list[str]):
        self.classes = classes
        self._models: list[IsotonicRegression] = []
        self._fitted = False

    def fit(self, P: np.ndarray, labels: pd.Series | np.ndarray) -> "IsotonicCalibrator":
        """
        Fit one calibrator per class.

        Parameters
        ----------
        P      : (N, K) uncalibrated posteriors
        labels : true string or integer labels
        """
        y = np.asarray(labels)
        Y = label_binarize(y, classes=self.classes)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        self._models = []
        for k in range(len(self.classes)):
            scores = P[:, k].clip(1e-6, 1 - 1e-6)
            y_k = Y[:, k] if Y.shape[1] > 1 else Y[:, 0]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(scores, y_k)
            self._models.append(iso)

        self._fitted = True
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        """Apply calibration and renormalize each row to sum to 1."""
        if not self._fitted:
            return P
        P_cal = np.zeros_like(P)
        for k, iso in enumerate(self._models):
            P_cal[:, k] = iso.predict(P[:, k].clip(1e-6, 1 - 1e-6))
        row_sums = P_cal.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return P_cal / row_sums

    def fit_transform(self, P: np.ndarray, labels) -> np.ndarray:
        return self.fit(P, labels).transform(P)


def ece_score(P: np.ndarray, labels: np.ndarray | pd.Series,
              classes: list[str], n_bins: int = 10) -> float:
    """
    Multiclass Expected Calibration Error (ECE).

    ECE = Σ_b (|B_b| / N) × |acc(B_b) − conf(B_b)|
    where B_b is the b-th confidence bin.
    """
    y = np.asarray(labels)
    confs = P.max(axis=1)
    preds = np.array([classes[i] for i in P.argmax(axis=1)])
    correct = (preds == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (confs >= a) & (confs < b + (1e-9 if b == 1.0 else 0))
        if mask.sum() > 0:
            ece += (mask.sum() / len(y)) * abs(confs[mask].mean() - correct[mask].mean())
    return float(ece)


def brier_score(P: np.ndarray, labels: np.ndarray | pd.Series,
                classes: list[str]) -> float:
    """
    Multiclass Brier score (mean of one-vs-rest Brier scores).

    A score of 0 is perfect; 1 is the worst case.
    """
    y = np.asarray(labels)
    Y = label_binarize(y, classes=classes)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return float(((P - Y) ** 2).mean())


def reliability_data(P: np.ndarray, labels: np.ndarray | pd.Series,
                     classes: list[str], n_bins: int = 10) -> pd.DataFrame:
    """
    Data for the reliability (calibration) diagram.

    Returns a DataFrame with columns [bin_center, accuracy, confidence, count].
    """
    y = np.asarray(labels)
    confs = P.max(axis=1)
    preds = np.array([classes[i] for i in P.argmax(axis=1)])
    correct = (preds == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (confs >= a) & (confs < b + (1e-9 if b == 1.0 else 0))
        if mask.sum() > 0:
            rows.append({
                "bin_center": (a + b) / 2,
                "accuracy":   correct[mask].mean(),
                "confidence": confs[mask].mean(),
                "count":      int(mask.sum()),
            })
    return pd.DataFrame(rows)
