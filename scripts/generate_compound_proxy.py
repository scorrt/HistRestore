"""Generate synthetic compound-degradation proxy pages.

This script implements the public generation protocol used for the
compound-proxy subset. It expects a directory of clean historical page images
and writes paired clean/degraded images plus a JSONL metadata manifest.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def resize_long_side(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    long_side = max(h, w)
    if max_side <= 0 or long_side <= max_side:
        return image
    scale = max_side / long_side
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def apply_perspective(image: np.ndarray, severity: float, rng: random.Random) -> np.ndarray:
    h, w = image.shape[:2]
    amp_x = 0.08 * severity * w
    amp_y = 0.08 * severity * h
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[x + rng.uniform(-amp_x, amp_x), y + rng.uniform(-amp_y, amp_y)] for x, y in src])
    mat = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)


def apply_shadow(image: np.ndarray, severity: float, rng: random.Random) -> np.ndarray:
    h, w = image.shape[:2]
    side = rng.choice(["left", "right", "top", "bottom", "spine"])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if side == "left":
        base = 1.0 - xx / max(w - 1, 1)
    elif side == "right":
        base = xx / max(w - 1, 1)
    elif side == "top":
        base = 1.0 - yy / max(h - 1, 1)
    elif side == "bottom":
        base = yy / max(h - 1, 1)
    else:
        base = 1.0 - np.abs(xx - 0.5 * w) / max(0.5 * w, 1)
    sigma = max(1.0, 0.03 * max(w, h))
    mask = cv2.GaussianBlur(np.clip(base, 0, 1), (0, 0), sigma)
    strength = 0.22 + 0.45 * severity
    shaded = image.astype(np.float32) * (1.0 - strength * mask[..., None])
    return np.clip(shaded, 0, 255).astype(np.uint8)


def apply_blur(image: np.ndarray, severity: float, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        k = int(3 + 10 * severity) | 1
        return cv2.GaussianBlur(image, (k, k), 0)
    k = int(5 + 18 * severity) | 1
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    angle = rng.uniform(0, 180)
    mat = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, mat, (k, k))
    kernel = kernel / max(kernel.sum(), 1e-6)
    return cv2.filter2D(image, -1, kernel)


def apply_yellowing(image: np.ndarray, severity: float, rng: random.Random) -> np.ndarray:
    contrast = 1.0 - (0.25 + 0.35 * severity)
    tint = np.array(
        [
            rng.uniform(8, 25),
            rng.uniform(18, 45),
            rng.uniform(28, 70),
        ],
        dtype=np.float32,
    )
    out = 127.5 + contrast * (image.astype(np.float32) - 127.5) + severity * tint
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_bleed_stain(image: np.ndarray, severity: float, rng: random.Random) -> np.ndarray:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bleed = 255 - gray
    shift_x = int(rng.uniform(-0.025, 0.025) * w)
    shift_y = int(rng.uniform(-0.025, 0.025) * h)
    mat = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    bleed = cv2.warpAffine(bleed, mat, (w, h), borderMode=cv2.BORDER_REFLECT)
    bleed = cv2.GaussianBlur(bleed, (0, 0), max(1.0, 0.008 * max(w, h)))
    alpha = 0.10 + 0.20 * severity
    out = image.astype(np.float32) - alpha * bleed[..., None]
    min_side = min(h, w)
    for _ in range(rng.randint(2, 7)):
        center = (rng.randrange(w), rng.randrange(h))
        radius = rng.randint(max(3, min_side // 80), max(4, min_side // 18))
        color = np.array([rng.uniform(60, 120), rng.uniform(70, 130), rng.uniform(90, 155)])
        layer = np.zeros_like(out)
        cv2.circle(layer, center, radius, color.tolist(), -1)
        stain = cv2.GaussianBlur(layer, (0, 0), radius / 3)
        out = np.where(stain > 0, 0.78 * out + 0.22 * stain, out)
    return np.clip(out, 0, 255).astype(np.uint8)


OPERATORS = {
    "perspective": apply_perspective,
    "shadow": apply_shadow,
    "blur": apply_blur,
    "low_contrast_yellowing": apply_yellowing,
    "bleed_through_stain": apply_bleed_stain,
}


def degrade(image: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict]:
    op_count = rng.choices([1, 2, 3], weights=[0.30, 0.50, 0.20], k=1)[0]
    names = rng.sample(list(OPERATORS), k=op_count)
    out = image.copy()
    records = []
    for name in names:
        severity = rng.uniform(0.25, 0.90)
        out = OPERATORS[name](out, severity, rng)
        records.append({"operator": name, "severity": round(severity, 6)})
    jpeg_quality = None
    if rng.random() < 0.35:
        jpeg_quality = rng.randint(35, 75)
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out, {"operators": records, "jpeg_quality": jpeg_quality}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    images = list_images(args.clean_root)
    if args.max_images:
        images = images[: args.max_images]

    clean_out = args.out_root / "clean"
    degraded_out = args.out_root / "degraded"
    clean_out.mkdir(parents=True, exist_ok=True)
    degraded_out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    metadata_path = args.out_root / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for page_idx, path in enumerate(images, start=1):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = resize_long_side(image, args.max_side)
            source_page_id = f"source_{page_idx:05d}"
            for variant_idx in range(args.variants):
                sample_id = f"{source_page_id}_v{variant_idx + 1:02d}"
                degraded, meta = degrade(image, rng)
                clean_name = f"{sample_id}_clean.png"
                degraded_name = f"{sample_id}_degraded.png"
                cv2.imwrite(str(clean_out / clean_name), image)
                cv2.imwrite(str(degraded_out / degraded_name), degraded)
                record = {
                    "sample_id": sample_id,
                    "source_page_id": source_page_id,
                    "source_file": path.name,
                    "clean_file": f"clean/{clean_name}",
                    "degraded_file": f"degraded/{degraded_name}",
                    "seed": args.seed,
                    **meta,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
