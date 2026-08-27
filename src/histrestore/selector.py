from collections import Counter
from typing import Dict, Iterable, List

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


class UtilitySelector:
    """Risk-aware page-wise utility selector."""

    def __init__(self, ssim_weight=8.0, risk_penalty=0.55, harm_penalty=0.15, random_state=202615):
        self.ssim_weight = ssim_weight
        self.risk_penalty = risk_penalty
        self.harm_penalty = harm_penalty
        self.model = GradientBoostingRegressor(
            n_estimators=280,
            max_depth=3,
            learning_rate=0.025,
            min_samples_leaf=6,
            random_state=random_state,
        )

    def target(self, row: Dict[str, float], base: Dict[str, float]) -> float:
        risk = float(row.get("content_risk", row.get("risk", 0.0)))
        base_risk = float(base.get("content_risk", base.get("risk", 0.0)))
        return (
            float(row["psnr"]) - float(base["psnr"])
            + self.ssim_weight * (float(row["ssim"]) - float(base["ssim"]))
            - self.risk_penalty * max(0.0, risk - base_risk)
            - self.harm_penalty * max(0.0, float(base["psnr"]) - float(row["psnr"]))
        )

    def fit(self, features: np.ndarray, targets: np.ndarray):
        self.model.fit(np.asarray(features, dtype=float), np.asarray(targets, dtype=float))
        return self

    def predict(self, features: np.ndarray):
        return self.model.predict(np.asarray(features, dtype=float))

    def select(self, candidate_names: List[str], features: np.ndarray):
        scores = self.predict(features)
        idx = int(np.argmax(scores))
        return candidate_names[idx], float(scores[idx])


def selected_frequency(rows: Iterable[Dict[str, str]]):
    return Counter(r["selected_candidate"] for r in rows)

