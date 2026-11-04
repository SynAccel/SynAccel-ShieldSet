"""Neighborhood label-consensus auditing for poisoned training data."""
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class AuditResult:
    flagged_indices: np.ndarray
    suspicion_scores: np.ndarray
    metrics: Dict[str, float]


class ShieldSetAuditor:
    """Flags samples whose label disagrees with their k nearest neighbors.

    The suspicion score of a sample is the fraction of its k neighbors
    (excluding itself) carrying a different label.
    """

    def __init__(self, k: int = 7, suspicion_threshold: float = 0.7):
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        if not 0.0 <= suspicion_threshold <= 1.0:
            raise ValueError("suspicion_threshold must be in [0, 1]")
        self.k = int(k)
        self.suspicion_threshold = float(suspicion_threshold)

    @staticmethod
    def _validate(X, y) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            X = X.reshape(len(X), -1)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1-D with the same length as X")
        if len(X) < 2:
            raise ValueError("at least 2 samples are required")
        return X, y

    def _neighbor_labels(self, X, y) -> np.ndarray:
        k = min(self.k, len(X) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        idx = nn.kneighbors(X, return_distance=False)
        # drop self explicitly (duplicates may make self non-first)
        rows = []
        for i, row in enumerate(idx):
            others = row[row != i]
            rows.append(others[:k])
        return y[np.array(rows)]

    def audit(self, X, y) -> AuditResult:
        X, y = self._validate(X, y)
        nl = self._neighbor_labels(X, y)
        scores = (nl != y[:, None]).mean(axis=1)
        flagged = np.flatnonzero(scores >= self.suspicion_threshold)
        metrics = {
            "n_samples": int(len(X)),
            "k": int(nl.shape[1]),
            "threshold": self.suspicion_threshold,
            "n_flagged": int(len(flagged)),
            "flagged_fraction": float(len(flagged) / len(X)),
            "mean_suspicion": float(scores.mean()),
            "max_suspicion": float(scores.max()),
        }
        return AuditResult(flagged, scores, metrics)

    def repair(self, X, y, strategy: str = "relabel"):
        """Return cleaned (X, y). strategy: 'relabel' or 'remove'."""
        if strategy not in ("relabel", "remove"):
            raise ValueError("strategy must be 'relabel' or 'remove'")
        X, y = self._validate(X, y)
        result = self.audit(X, y)
        flagged = result.flagged_indices
        if strategy == "remove":
            keep = np.ones(len(X), dtype=bool)
            keep[flagged] = False
            return X[keep], y[keep]
        y_new = y.copy()
        if len(flagged):
            nl = self._neighbor_labels(X, y)
            for i in flagged:
                vals, counts = np.unique(nl[i], return_counts=True)
                y_new[i] = vals[np.argmax(counts)]
        return X, y_new
