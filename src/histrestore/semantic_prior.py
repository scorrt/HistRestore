from dataclasses import dataclass, field
from typing import Dict, Iterable

import numpy as np

from .candidate_bank import candidate_strength, infer_family


@dataclass
class SemanticPrior:
    """Distilled semantic prior over restoration candidates."""

    probabilities: Dict[str, float] = field(default_factory=dict)

    def normalized(self, methods: Iterable[str]):
        vals = np.asarray([self.probabilities.get(m, 0.0) for m in methods], dtype=float)
        if vals.sum() <= 0:
            vals[:] = 1.0 / max(len(vals), 1)
        else:
            vals /= vals.sum()
        return dict(zip(methods, vals.tolist()))


def compatibility_features(candidate: str, prior: SemanticPrior, methods):
    """Candidate-aware semantic compatibility features.

    These frozen features convert a page-level semantic prior into
    candidate-specific evidence before utility estimation.
    """

    q = prior.normalized(methods)
    cur = float(q.get(candidate, 0.0))
    fam = infer_family(candidate)
    fam_sum = float(sum(q.get(m, 0.0) for m in methods if infer_family(m) == fam))
    top = max(methods, key=lambda m: q.get(m, 0.0))
    vals = np.asarray([q.get(m, 0.0) for m in methods], dtype=float)
    entropy = float(-(vals * np.log(vals + 1e-12)).sum())
    confidence = 1.0 - entropy / max(np.log(len(methods)), 1e-12)
    target_strength = float(sum(q.get(m, 0.0) * candidate_strength(m) for m in methods))
    c_strength = candidate_strength(candidate)
    return {
        "prior_candidate": cur,
        "prior_family": fam_sum,
        "top_candidate_match": float(candidate == top),
        "top_family_match": float(fam == infer_family(top)),
        "strength_gap": abs(c_strength - target_strength),
        "prior_confidence": confidence,
        "prior_entropy": entropy,
    }

