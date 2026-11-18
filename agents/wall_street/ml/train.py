"""Train the price-move model end to end.

    python -m agents.wall_street.ml.train                     # defaults: 5-day horizon, beat-SPY target
    python -m agents.wall_street.ml.train --target direction  # predict up/down instead
    python -m agents.wall_street.ml.train --refresh           # re-download prices
    # Experiment 1: cross-sectional ranks + "beat the median stock" + monthly horizon
    python -m agents.wall_street.ml.train --features rank --target xs --horizon 21 --no-save

Steps:
  1. Build the dataset (downloads ~10 years of daily prices; cached in data/raw/).
  2. Walk-forward validate every candidate model, including the naive baseline. The best
     model is picked by rank IC (does a higher score really mean a better stock?).
  3. Backtest the best model's out-of-sample signals two ways: per-stock threshold timing,
     and holding the top 20% of stocks - both against holding everything.
  4. Refit the best model on ALL data (more data = better final model) and save it with
     the out-of-sample metrics from step 2 - never metrics measured on training data.
     (--no-save stops after step 3, so experiments don't replace the agent's model.)
"""

from datetime import date
import argparse
import logging

import pandas as pd

from core.config import settings
from agents.wall_street.data.providers import YFinanceProvider
from agents.wall_street.ml.backtest import Backtester
from agents.wall_street.ml.dataset import DEFAULT_UNIVERSE, TARGETS, DatasetBuilder, split_xy
from agents.wall_street.ml.features import FEATURE_SETS
from agents.wall_street.ml.models import MODEL_CLASSES
from agents.wall_street.ml.registry import ModelRegistry
from agents.wall_street.ml.trainer import Trainer

log = logging.getLogger("train")

# A model must clear BOTH bars to be called useful: beat a coin flip by a little AUC, and
# have ranking skill that's statistically distinguishable from luck (t > 2 means a result
# this strong would happen by chance only ~5% of the time).
MIN_USEFUL_AUC = 0.51
MIN_IC_TSTAT = 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the WallStreetAgent price-move model.")
    parser.add_argument("--horizon", type=int, default=5, help="days ahead to predict")
    parser.add_argument("--target", choices=TARGETS, default="excess")
    parser.add_argument("--features", choices=list(FEATURE_SETS), default="raw",
                        help="raw per-stock values, or cross-sectional ranks per date")
    parser.add_argument("--splits", type=int, default=5, help="walk-forward folds")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_UNIVERSE)
    parser.add_argument("--refresh", action="store_true", help="re-download prices")
    parser.add_argument("--no-save", action="store_true",
                        help="evaluate only; don't replace the model the agent uses")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("yfinance", "urllib3", "peewee"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    pd.set_option("display.width", 200)

    # 1. Data -----------------------------------------------------------------------
    log.info("Building dataset (%d tickers, horizon=%dd, target=%s, features=%s)...",
             len(args.tickers), args.horizon, args.target, args.features)
    columns = FEATURE_SETS[args.features]
    builder = DatasetBuilder(YFinanceProvider(), settings.data_dir / "raw")
    data = builder.build(args.tickers, horizon=args.horizon, target=args.target, refresh=args.refresh)
    data = data.dropna(subset=columns)   # rank features need >= MIN_CROSS_SECTION stocks per date
    log.info("Dataset: %d rows, %d tickers, %s -> %s, label base rate %.1f%%\n",
             len(data), data["ticker"].nunique(), data["date"].min().date(),
             data["date"].max().date(), data["label"].mean() * 100)

    # 2. Walk-forward validation ------------------------------------------------------
    trainer = Trainer(n_splits=args.splits, horizon=args.horizon)
    reports = {}
    for name, cls in MODEL_CLASSES.items():
        log.info("Walk-forward: %s", name)
        reports[name] = trainer.walk_forward(cls(), data, columns)

    summary = pd.DataFrame([r.summary() for r in reports.values()]).set_index("model")
    log.info("\nOut-of-sample results (averaged over %d folds):\n%s\n", args.splits, summary.to_string())
    best_name = summary.drop(index="naive_baseline")["rank_ic"].idxmax()
    best = reports[best_name]
    log.info("Per-fold results for %s:\n%s\n", best_name,
             pd.DataFrame(best.folds).set_index("fold").round(4).to_string())
    q = best.quantiles()
    log.info("Average %d-day forward return by score quintile (Q1 = lowest scores): %s",
             args.horizon, q["bucket_mean_fwd_return_pct"])
    log.info("Top minus bottom: %.3f%% per period, positive in %.1f%% of periods, monotonic: %s\n",
             q["top_minus_bottom_pct"], q["top_beats_bottom_pct_of_periods"], q["monotonic"])

    # 3. Backtests ------------------------------------------------------------------
    bt = Backtester(threshold=0.5, cost_bps=5).run(best.oos)
    log.info("Backtest A - hold a stock when prob > 0.5 (%s -> %s, %d days, 5 bps per trade):",
             bt.start, bt.end, bt.days)
    log.info("%s\n", pd.DataFrame({"strategy": bt.strategy, "buy_and_hold": bt.benchmark}).to_string())
    top = Backtester(cost_bps=5).run_top_bucket(best.oos, rebalance_every=args.horizon)
    log.info("Backtest B - hold the top 20%% by score, rebalance every %d days:", args.horizon)
    log.info("%s\n", pd.DataFrame({"top_20pct": top.strategy, "hold_all": top.benchmark}).to_string())

    # 4. Final fit + save -----------------------------------------------------------
    best_summary = best.summary()
    beats_baseline = best_summary["auc"] >= MIN_USEFUL_AUC and best_summary["ic_tstat"] >= MIN_IC_TSTAT
    log.info("Edge check for %s: AUC %.4f (need %.2f), IC t-stat %.2f (need %.1f) -> %s",
             best_name, best_summary["auc"], MIN_USEFUL_AUC, best_summary["ic_tstat"], MIN_IC_TSTAT,
             "REAL EDGE" if beats_baseline else "no reliable edge")
    if args.no_save:
        log.info("--no-save: evaluation only, the agent keeps its current model.")
        return

    X, y = split_xy(data, columns)
    final = MODEL_CLASSES[best_name]().fit(X, y)
    version = ModelRegistry(settings.models_dir).save(final, {
        "trained_on": date.today().isoformat(),
        "horizon_days": args.horizon,
        "target": args.target,
        "feature_set": args.features,
        "features": columns,
        "universe": sorted(data["ticker"].unique()),
        "data_start": data["date"].min().date(),
        "data_end": data["date"].max().date(),
        "label_base_rate": round(float(data["label"].mean()), 4),
        "oos_metrics": best_summary,
        "all_models": summary.reset_index().to_dict(orient="records"),
        "quantiles": q,
        "backtest": {"strategy": bt.strategy, "buy_and_hold": bt.benchmark,
                     "start": bt.start, "end": bt.end},
        "backtest_top_bucket": {"strategy": top.strategy, "hold_all": top.benchmark},
        "beats_baseline": bool(beats_baseline),
    })
    log.info("Saved %s as models/price_move/%s", best_name, version)
    if not beats_baseline:
        log.info("NOTE: the agent will be told this model has no reliable edge and should "
                 "give its signal little weight.")


if __name__ == "__main__":
    main()
