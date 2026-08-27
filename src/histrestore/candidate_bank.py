from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass
class CandidateRecord:
    """One restoration candidate and its evidence/metric metadata."""

    sample_id: str
    name: str
    image: Optional[np.ndarray] = None
    evidence: Dict[str, float] = field(default_factory=dict)
    family: str = ""


class CandidateBank:
    """A page-wise restoration candidate bank.

    HistRestore treats external restorers, classical operations, blends, and
    region-aware candidates as candidate sources. The selection mechanism is
    evaluated over this bank rather than tied to one restoration backbone.
    """

    def __init__(self, sample_id: str):
        self.sample_id = sample_id
        self._records: Dict[str, CandidateRecord] = {}

    def add(self, name: str, image=None, evidence=None, family: str = "") -> None:
        self._records[name] = CandidateRecord(
            sample_id=self.sample_id,
            name=name,
            image=image,
            evidence=dict(evidence or {}),
            family=family or infer_family(name),
        )

    def get(self, name: str) -> CandidateRecord:
        return self._records[name]

    def names(self) -> List[str]:
        return list(self._records)

    def values(self) -> Iterable[CandidateRecord]:
        return self._records.values()

    def __contains__(self, name: str) -> bool:
        return name in self._records


def infer_family(name: str) -> str:
    if name in {"input", "input_degraded"}:
        return "input"
    if name.startswith("classical") or name.startswith("shadow"):
        return "classical"
    if name == "mmdir_official":
        return "mmdir"
    if "region" in name:
        return "region"
    if "blend" in name:
        return "blend"
    if name.startswith("docres"):
        return "docres"
    if "binary" in name or "binar" in name:
        return "binarization"
    if "appearance" in name:
        return "appearance"
    return "other"


def candidate_strength(name: str) -> float:
    family = infer_family(name)
    if family == "input":
        return 0.0
    if family == "classical":
        return 0.4
    if "0.80" in name:
        return 0.8
    if "0.90" in name or "0.95" in name:
        return 0.9
    if family == "mmdir":
        return 1.0
    if family in {"docres", "region", "appearance", "binarization"}:
        return 0.85
    return 0.5

