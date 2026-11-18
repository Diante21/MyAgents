"""Walk-forward validation: the only honest way to evaluate a time-series model.

    |---- train ----|gap|-- test 1 --|
    |-------- train --------|gap|-- test 2 --|
    |------------ train ------------|gap|-- test 3 --|

* Never shuffle. A random train/test split puts tomorrow's data in training, which
  leaks the future into the model.
* The GAP ("embargo"): a label at day t looks `horizon` days ahead. Training rows in the
  last `horizon` days before the test window have labels that overlap the test period,
  so they're dropped.
* Every prediction in the stitched-together test windows was made by a model that only
  saw the past - exactly the situation you'd be in live. These "out-of-sample" (OOS)
  predictions feed the backtest.
"""

from dataclasses import dataclass, field
import copy

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from agents.wall_street.ml.dataset import split_xy
from agents.wall_street.ml.features import FEATURE_COLUMNS
from agents.wall_street.ml.metrics import daily_rank_ic, quantile_spread, rank_ic_summary
from agents.wall_street.ml.models import BasePredictor


@dataclass
class WalkForwardReport:
    model_name: str
    horizon: int = 5
    folds: list[dict] = field(default_factory=list)
    oos: pd.DataFrame | None = None   # date, ticker, prob, label, returns - for the backtest

    def summary(self) -> dict:
        df = pd.DataFrame(self.folds)
        return {
            "model": self.model_name,
            "folds": len(df),
            "accuracy": round(df["accuracy"].mean(), 4),
            "baseline_accuracy": round(df["baseline_accuracy"].mean(), 4),
            "auc": round(df["auc"].mean(), 4),
            "auc_min_fold": round(df["auc"].min(), 4),
            "log_loss": round(df["log_loss"].mean(), 4),
            # Ranking quality over ALL out-of-sample predictions (see metrics.py).
            **rank_ic_summary(self.oos, self.horizon),
            "top_minus_bottom_pct": quantile_spread(self.oos, self.horizon).get("top_minus_bottom_pct", 0.0),
        }

    def quantiles(self) -> dict:
        return quantile_spread(self.oos, self.horizon)


class Trainer:
    def __init__(self, n_splits: int = 5, initial_train_frac: float = 0.4, horizon: int = 5):
        self.n_splits = n_splits
        self.initial_train_frac = initial_train_frac
        self.horizon = horizon
        self.embargo = horizon

    def folds(self, dates: pd.Series):
        """Yield (train_dates, test_dates) pairs, oldest first."""
        unique = np.sort(dates.unique())
        start = int(len(unique) * self.initial_train_frac)
        edges = np.linspace(start, len(unique), self.n_splits + 1).astype(int)
        for lo, hi in zip(edges[:-1], edges[1:]):
            train_end = max(lo - self.embargo, 1)
            yield unique[:train_end], unique[lo:hi]

    def walk_forward(self, model: BasePredictor, data: pd.DataFrame,
                     columns: list[str] = FEATURE_COLUMNS) -> WalkForwardReport:
        report = WalkForwardReport(model_name=model.name, horizon=self.horizon)
        oos_parts = []
        for i, (train_dates, test_dates) in enumerate(self.folds(data["date"]), start=1):
            train = data[data["date"].isin(train_dates)]
            test = data[data["date"].isin(test_dates)]
            X_tr, y_tr = split_xy(train, columns)
            X_te, y_te = split_xy(test, columns)

            fitted = copy.deepcopy(model).fit(X_tr, y_tr)   # fresh copy per fold
            prob = fitted.predict_proba(X_te)
            majority = int(y_tr.mean() >= 0.5)

            report.folds.append({
                "fold": i,
                "train_rows": len(train),
                "test_start": str(pd.Timestamp(test_dates[0]).date()),
                "test_end": str(pd.Timestamp(test_dates[-1]).date()),
                "accuracy": accuracy_score(y_te, prob >= 0.5),
                "baseline_accuracy": accuracy_score(y_te, np.full(len(y_te), majority)),
                "auc": roc_auc_score(y_te, prob) if y_te.nunique() > 1 else 0.5,
                "log_loss": log_loss(y_te, np.clip(prob, 1e-6, 1 - 1e-6), labels=[0, 1]),
            })
            fold_oos = test[["date", "ticker", "label", "fwd_return", "next_day_return"]].assign(prob=prob)
            report.folds[-1]["rank_ic"] = float(daily_rank_ic(fold_oos).mean())
            oos_parts.append(fold_oos)

        report.oos = pd.concat(oos_parts, ignore_index=True)
        return report
