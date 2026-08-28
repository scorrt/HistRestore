from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class EvidenceVector:
    edge_jaccard: float
    edge_keep: float
    foreground_shift: float
    mean_shift: float
    contrast_shift: float
    contrast_after: float
    sharp_after: float
    content_risk: float

    def to_dict(self):
        return asdict(self)


def _gray(image):
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _edge_mask(gray):
    edges = cv2.Canny(gray, 80, 160)
    return edges > 0


def _foreground_mask(gray):
    # Robust foreground proxy for historical document pages.
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    return gray < np.percentile(blur, 58)


def _rho(value, scale=1.0):
    """Saturating normalizer used by VCCRP risk terms."""

    return np.clip(np.asarray(value, dtype=np.float32) / max(scale, 1e-12), 0.0, 1.0)


def extract_pair_evidence(source, candidate) -> EvidenceVector:
    """Compute source-relative no-reference evidence for one candidate."""

    src = _gray(source).astype(np.float32)
    cand = _gray(candidate).astype(np.float32)
    if src.shape != cand.shape:
        cand = cv2.resize(cand, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_AREA)

    e_src = _edge_mask(src.astype(np.uint8))
    e_cand = _edge_mask(cand.astype(np.uint8))
    inter = np.logical_and(e_src, e_cand).sum()
    union = np.logical_or(e_src, e_cand).sum()
    edge_jaccard = inter / max(union, 1)
    edge_keep = inter / max(e_src.sum(), 1)

    f_src = _foreground_mask(src)
    f_cand = _foreground_mask(cand)
    foreground_shift = abs(float(f_cand.mean()) - float(f_src.mean()))
    mean_shift = abs(float(cand.mean()) - float(src.mean())) / 255.0
    contrast_before = float(src.std())
    contrast_after = float(cand.std())
    contrast_shift = abs(contrast_after - contrast_before) / 255.0
    sharp_after = float(cv2.Laplacian(cand, cv2.CV_32F).var())

    edge_discontinuity = 1.0 - edge_jaccard
    content_risk = (
        0.45 * float(_rho(edge_discontinuity))
        + 0.25 * float(_rho(foreground_shift))
        + 0.20 * float(_rho(contrast_shift))
        + 0.10 * float(_rho(mean_shift))
    )

    return EvidenceVector(
        edge_jaccard=float(edge_jaccard),
        edge_keep=float(edge_keep),
        foreground_shift=float(foreground_shift),
        mean_shift=float(mean_shift),
        contrast_shift=float(contrast_shift),
        contrast_after=float(contrast_after),
        sharp_after=float(sharp_after),
        content_risk=float(content_risk),
    )
