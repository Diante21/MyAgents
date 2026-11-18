"""Price-move prediction: the part of WallStreetAgent you train yourself.

Pipeline (each step is one module):
    dataset.py   download prices -> features.py build features -> make labels
    models.py    candidate models (always including a naive baseline)
    trainer.py   walk-forward validation: train on the past, test on the future
    backtest.py  turn out-of-sample predictions into a trading P&L, with costs
    registry.py  save the final model + its honest metrics
    train.py     the script that runs all of the above

At run time, tools/predict.py loads the saved model and calls the SAME FeatureBuilder,
so features are computed identically in training and in production.
"""
