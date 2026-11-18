"""ModelRegistry: saves trained models with their metadata, and loads the latest one.

Layout:
    models/price_move/
        20261115-142530/
            model.joblib   the fitted predictor
            meta.json      features, horizon, target, universe, honest OOS metrics
        latest.txt         name of the newest version folder

Why store metadata next to the model? Because "a model" is useless without knowing what
it predicts, which features (in which order) it expects, and how good it really was.
The prediction tool passes those metrics to Claude so it can weigh the signal.

Security note: joblib files are pickles - loading one can run arbitrary code. Only load
models you trained yourself; never load model files downloaded from the internet.
"""

from datetime import datetime
from pathlib import Path
from typing import Any
import json

import joblib

from agents.wall_street.ml.models import BasePredictor


class ModelRegistry:
    def __init__(self, root: Path, name: str = "price_move"):
        self.dir = root / name

    def save(self, model: BasePredictor, meta: dict[str, Any]) -> str:
        version = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder = self.dir / version
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, folder / "model.joblib")
        meta = {"version": version, "model": model.name, **meta}
        (folder / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        (self.dir / "latest.txt").write_text(version, encoding="utf-8")
        return version

    def latest_version(self) -> str | None:
        pointer = self.dir / "latest.txt"
        return pointer.read_text(encoding="utf-8").strip() if pointer.exists() else None

    def load_latest(self) -> tuple[BasePredictor, dict[str, Any]]:
        version = self.latest_version()
        if version is None:
            raise FileNotFoundError("No trained model yet. Run: python -m agents.wall_street.ml.train")
        folder = self.dir / version
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        return joblib.load(folder / "model.joblib"), meta
