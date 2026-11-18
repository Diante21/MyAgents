"""Candidate models. All share one interface (BasePredictor) so the trainer, registry
and prediction tool don't care which one won.

Always train a NAIVE BASELINE alongside real models. In markets most "skill" turns out
to be the base rate in disguise; a model only earns its place if it beats the baseline
out of sample.

Why these models and not deep learning first?
  * Stock features are tabular and extremely noisy (signal-to-noise is tiny).
  * Gradient-boosted trees are the standard strong baseline for tabular data.
  * Heavy regularisation (shallow trees, big leaves) matters more than model size here:
    a flexible model will happily memorise noise.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class BasePredictor(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BasePredictor": ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of label 1 for each row (1-D array)."""


class NaiveBaselinePredictor(BasePredictor):
    """Ignores the features and always predicts the training base rate.
    Its accuracy = 'always guess the most common class'. Its AUC is exactly 0.5."""
    name = "naive_baseline"

    def fit(self, X, y):
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.full(len(X), self.rate_)


class LogisticPredictor(BasePredictor):
    """Linear model. Strong regularisation (small C) because the signal is weak.
    Scaling first matters: logistic regression is sensitive to feature scale."""
    name = "logistic"

    def __init__(self, C: float = 0.05):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class GradientBoostPredictor(BasePredictor):
    """scikit-learn's histogram gradient boosting (same family as LightGBM).
    Can capture non-linear effects like 'momentum works, except when volatility spikes'."""
    name = "gradient_boost"

    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            max_depth=3,             # shallow trees = simple interactions only
            learning_rate=0.03,
            max_iter=250,
            min_samples_leaf=500,    # each leaf must cover many days -> ignores one-off noise
            l2_regularization=1.0,
            early_stopping=False,    # early stopping would use a RANDOM split = leakage in time series
            random_state=42,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


MODEL_CLASSES: dict[str, type[BasePredictor]] = {
    cls.name: cls for cls in (NaiveBaselinePredictor, LogisticPredictor, GradientBoostPredictor)
}
