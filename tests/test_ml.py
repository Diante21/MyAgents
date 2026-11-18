"""Tests for the ML pipeline. The leakage tests are the most important tests in the
project: a leak silently makes every metric meaningless."""

import numpy as np
import pandas as pd
import pytest

from agents.wall_street import WallStreetAgent
from agents.wall_street.ml.backtest import Backtester
from agents.wall_street.ml.dataset import DatasetBuilder
from agents.wall_street.ml.features import FEATURE_COLUMNS, FeatureBuilder
from agents.wall_street.ml.models import GradientBoostPredictor, LogisticPredictor, NaiveBaselinePredictor
from agents.wall_street.ml.registry import ModelRegistry
from agents.wall_street.ml.trainer import Trainer
from agents.wall_street.tools.predict import PredictPriceMoveTool
from tests.conftest import FakeLLM, FakeProvider


def random_walk(days=600, seed=0, start="2022-01-03"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=days)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, days))), index=idx)
    volume = pd.Series(rng.integers(1_000_000, 5_000_000, days).astype(float), index=idx)
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": volume})


class RandomProvider(FakeProvider):
    """Different random-walk prices per ticker."""
    def get_price_history(self, ticker, period="1y", interval="1d"):
        return random_walk(seed=sum(map(ord, ticker)))


# ---- leakage ------------------------------------------------------------------------
def test_features_do_not_use_the_future():
    prices, market = random_walk(seed=1), random_walk(seed=2)
    before = FeatureBuilder().build(prices, market)

    cut = prices.index[450]
    future_changed = prices.copy()
    future_changed.loc[future_changed.index > cut, ["Close", "Volume"]] *= 3   # rewrite the future
    after = FeatureBuilder().build(future_changed, market)

    past = before.index[before.index <= cut]
    pd.testing.assert_frame_equal(before.loc[past], after.loc[past])


def test_labels_look_forward_exactly_horizon_days():
    close = pd.Series([100, 101, 99, 105, 104, 110], dtype=float,
                      index=pd.bdate_range("2026-01-05", periods=6))
    label = DatasetBuilder.make_labels(close, close, horizon=2, target="direction")
    assert label.tolist()[:4] == [0.0, 1.0, 1.0, 1.0]     # 99<100, 105>101, 104>99, 110>105
    assert label.iloc[-2:].isna().all()                   # future unknown -> no label


def test_walk_forward_trains_only_on_the_past_with_a_gap():
    dates = pd.Series(pd.bdate_range("2020-01-01", periods=500))
    trainer = Trainer(n_splits=4, horizon=5)
    for train, test in trainer.folds(dates):
        gap = np.busday_count(pd.Timestamp(train[-1]).date(), pd.Timestamp(test[0]).date())
        assert gap > 5


# ---- models + pipeline ---------------------------------------------------------------
@pytest.fixture
def dataset(tmp_path):
    return DatasetBuilder(RandomProvider(), tmp_path / "raw").build(["AAA", "BBB", "CCC"], horizon=5)


def test_dataset_has_features_and_no_label_leaks(dataset):
    assert set(FEATURE_COLUMNS) <= set(dataset.columns)
    assert dataset[FEATURE_COLUMNS].notna().all().all()
    assert "label" not in FEATURE_COLUMNS and "fwd_return" not in FEATURE_COLUMNS


@pytest.mark.parametrize("cls", [NaiveBaselinePredictor, LogisticPredictor, GradientBoostPredictor])
def test_models_output_probabilities(cls, dataset):
    X, y = dataset[FEATURE_COLUMNS], dataset["label"].astype(int)
    prob = cls().fit(X, y).predict_proba(X)
    assert prob.shape == (len(X),) and ((prob >= 0) & (prob <= 1)).all()


def test_random_walk_has_no_edge(dataset):
    """On pure noise, an honest pipeline must NOT find skill. If this ever fails with a
    high AUC, something is leaking."""
    report = Trainer(n_splits=3, horizon=5).walk_forward(LogisticPredictor(), dataset)
    assert 0.40 < report.summary()["auc"] < 0.60


def test_backtest_charges_costs(dataset):
    oos = dataset[["date", "ticker", "label", "fwd_return", "next_day_return"]].assign(prob=0.9)
    free = Backtester(cost_bps=0).run(oos)
    costly = Backtester(cost_bps=50).run(oos)
    assert costly.strategy["total_return_pct"] < free.strategy["total_return_pct"]
    # Always invested + no costs = identical to buy-and-hold.
    assert free.strategy["total_return_pct"] == pytest.approx(free.benchmark["total_return_pct"])


# ---- registry + tool -----------------------------------------------------------------
def train_and_save(dataset, root):
    X, y = dataset[FEATURE_COLUMNS], dataset["label"].astype(int)
    registry = ModelRegistry(root)
    registry.save(LogisticPredictor().fit(X, y), {
        "trained_on": "2026-11-15", "horizon_days": 5, "target": "excess",
        "features": FEATURE_COLUMNS, "universe": ["AAA"], "beats_baseline": False,
        "oos_metrics": {"accuracy": 0.5, "baseline_accuracy": 0.5, "auc": 0.5},
        "backtest": {"strategy": {"sharpe": 0.1}, "buy_and_hold": {"sharpe": 0.2}},
    })
    return registry


def test_predict_tool_returns_probability_and_track_record(dataset, tmp_path):
    registry = train_and_save(dataset, tmp_path / "models")
    out = PredictPriceMoveTool(RandomProvider(), registry).run("AAA")
    assert 0 <= out["probability"] <= 1
    assert out["model"]["out_of_sample_auc"] == 0.5
    assert "outperforms" in out["what_probability_means"].lower()


def test_agent_offers_predict_tool_only_when_a_model_exists(dataset, tmp_path):
    empty = ModelRegistry(tmp_path / "empty")
    agent = WallStreetAgent(llm=FakeLLM([]), provider=RandomProvider(), model_registry=empty)
    assert "predict_price_move" not in agent.tools.names

    trained = train_and_save(dataset, tmp_path / "models")
    agent = WallStreetAgent(llm=FakeLLM([]), provider=RandomProvider(), model_registry=trained)
    assert "predict_price_move" in agent.tools.names


# ---- experiment 1: cross-sectional ranks, "xs" target, ranking metrics ---------------
from agents.wall_street.ml.features import RANK_COLUMNS, build_panel  # noqa: E402
from agents.wall_street.ml.metrics import quantile_spread, rank_ic_summary  # noqa: E402

UNIVERSE = list("ABCDEFGHIJKL")   # 12 tickers -> enough for a cross-section (distinct seeds)


@pytest.fixture
def rank_dataset(tmp_path):
    return DatasetBuilder(RandomProvider(), tmp_path / "raw").build(UNIVERSE, horizon=5, target="xs")


def test_rank_features_are_centered_percentiles(rank_dataset):
    ranks = rank_dataset[RANK_COLUMNS[:-3]]           # last 3 are raw market features
    assert ranks.notna().all().all()
    assert ranks.min().min() >= -0.5 and ranks.max().max() <= 0.5
    assert abs(rank_dataset.groupby("date")["ret_21d_xrank"].mean()).max() < 0.05


def test_ranks_do_not_use_the_future():
    prices = {t: random_walk(seed=i) for i, t in enumerate(UNIVERSE)}
    market = random_walk(seed=99)
    before = build_panel(FeatureBuilder(), prices, market)

    cut = prices["A"].index[450]
    changed = dict(prices)
    changed["A"] = prices["A"].copy()
    changed["A"].loc[changed["A"].index > cut, ["Close", "Volume"]] *= 3
    after = build_panel(FeatureBuilder(), changed, market)

    past = lambda p: p[p["date"] <= cut].reset_index(drop=True)   # noqa: E731
    pd.testing.assert_frame_equal(past(before), past(after))


def test_xs_label_splits_every_day_in_half(rank_dataset):
    assert (rank_dataset.groupby("date")["label"].mean() == 0.5).all()
    row = rank_dataset.iloc[0]
    same_day = rank_dataset[rank_dataset["date"] == row["date"]]
    assert row["label"] == float(row["fwd_return"] > same_day["fwd_return"].median())


def test_rank_ic_perfect_and_useless_predictors(rank_dataset):
    perfect = rank_dataset.assign(prob=rank_dataset["fwd_return"])
    s = rank_ic_summary(perfect, horizon=5)
    assert s["rank_ic"] == pytest.approx(1.0)
    # The t-stat only counts non-overlapping periods: every 5th date.
    assert s["ic_periods"] == int(np.ceil(rank_dataset["date"].nunique() / 5))
    q = quantile_spread(perfect, horizon=5)
    assert q["monotonic"] and q["top_minus_bottom_pct"] > 0

    rng = np.random.default_rng(0)
    noise = rank_dataset.assign(prob=rng.random(len(rank_dataset)))
    assert abs(rank_ic_summary(noise, horizon=5)["rank_ic"]) < 0.05


def test_top_bucket_backtest_rewards_skill_and_charges_costs(rank_dataset):
    foresight = rank_dataset.assign(prob=rank_dataset["next_day_return"])   # cheating on purpose
    free = Backtester(cost_bps=0).run_top_bucket(foresight, rebalance_every=1)
    costly = Backtester(cost_bps=50).run_top_bucket(foresight, rebalance_every=1)
    assert free.strategy["total_return_pct"] > free.benchmark["total_return_pct"]
    assert costly.strategy["total_return_pct"] < free.strategy["total_return_pct"]


def test_predict_tool_with_rank_model(rank_dataset, tmp_path):
    X, y = rank_dataset[RANK_COLUMNS], rank_dataset["label"].astype(int)
    registry = ModelRegistry(tmp_path / "models")
    registry.save(LogisticPredictor().fit(X, y), {
        "trained_on": "2026-11-16", "horizon_days": 5, "target": "xs", "feature_set": "rank",
        "features": RANK_COLUMNS, "universe": UNIVERSE, "beats_baseline": False,
        "oos_metrics": {"accuracy": 0.5, "baseline_accuracy": 0.5, "auc": 0.5,
                        "rank_ic": 0.01, "ic_tstat": 0.4},
        "backtest": {"strategy": {"sharpe": 0.1}, "buy_and_hold": {"sharpe": 0.2}},
    })
    out = PredictPriceMoveTool(RandomProvider(), registry).run("C")
    assert 0 <= out["probability"] <= 1
    assert out["model"]["rank_ic_t_stat"] == 0.4
    assert "median stock" in out["what_probability_means"].lower()


def test_constant_scores_have_no_quantile_spread(rank_dataset):
    assert quantile_spread(rank_dataset.assign(prob=0.5), horizon=5) == {}
