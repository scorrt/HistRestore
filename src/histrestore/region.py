import cv2
import numpy as np


def shadow_map(degraded):
    gray = degraded if degraded.ndim == 2 else cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    smooth = cv2.GaussianBlur(gray, (0, 0), 15.0)
    norm = (smooth - smooth.min()) / max(float(smooth.max() - smooth.min()), 1e-6)
    return 1.0 - norm


def content_map(degraded):
    gray = degraded if degraded.ndim == 2 else cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    ink = 1.0 - cv2.GaussianBlur(gray, (0, 0), 2.0)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.GaussianBlur(np.sqrt(sx * sx + sy * sy), (0, 0), 1.2)
    edge = edge / max(float(edge.max()), 1e-6)
    return np.clip(0.65 * ink + 0.45 * edge, 0.0, 1.0)


def alpha_map(degraded, alpha_min=0.68, alpha_max=0.98):
    s = shadow_map(degraded)
    c = content_map(degraded)
    alpha_raw = 0.82 + 0.18 * s * (1.0 - c) - 0.10 * c
    alpha = np.clip(alpha_raw, alpha_min, alpha_max)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 3.0)
    return alpha.astype(np.float32)


def build_region_candidate(degraded, base_restored):
    """Create a region-aware residual candidate.

    y_R = I_d + alpha * (I_D - I_d)
    """

    if degraded.shape != base_restored.shape:
        base_restored = cv2.resize(base_restored, (degraded.shape[1], degraded.shape[0]), interpolation=cv2.INTER_AREA)
    alpha = alpha_map(degraded)
    if degraded.ndim == 3:
        alpha = alpha[..., None]
    y = degraded.astype(np.float32) + alpha * (base_restored.astype(np.float32) - degraded.astype(np.float32))
    return np.clip(y, 0, 255).astype(np.uint8), alpha.squeeze()
