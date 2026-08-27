import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from run_baselines import psnr, ssim_gray


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def to_gray_binary(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (gray > 155).astype(np.uint8)


def binary_consistency(gt, before, after):
    gt_b = to_gray_binary(gt)
    b_b = to_gray_binary(before)
    a_b = to_gray_binary(after)
    before_correct = b_b == gt_b
    before_wrong = ~before_correct
    after_wrong = a_b != gt_b
    after_correct = a_b == gt_b
    correct_damage = float(np.logical_and(before_correct, after_wrong).sum() / max(before_correct.sum(), 1))
    wrong_repair = float(np.logical_and(before_wrong, after_correct).sum() / max(before_wrong.sum(), 1))
    net_gain = wrong_repair - correct_damage
    return {
        "correct_damage": correct_damage,
        "wrong_repair": wrong_repair,
        "net_gain": net_gain,
    }


def edge_consistency(before, after):
    b = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    a = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    eb = cv2.Canny(b, 80, 160)
    ea = cv2.Canny(a, 80, 160)
    inter = np.logical_and(eb > 0, ea > 0).sum()
    denom = max((eb > 0).sum(), 1)
    return float(inter / denom)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-dataset", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--binary", action="store_true")
    args = parser.parse_args()

    paired = Path(args.paired_dataset)
    pred_root = Path(args.prediction_root)
    rows = list(read_jsonl(paired / "metadata.jsonl"))
    metrics = []
    for rec in rows:
        clean = cv2.imread(str(paired / rec["clean"]), cv2.IMREAD_COLOR)
        before = cv2.imread(str(paired / rec["degraded"]), cv2.IMREAD_COLOR)
        pred_candidates = list(pred_root.glob(f"{rec['sample_id']}*"))
        pred_candidates = [p for p in pred_candidates if p.is_file()]
        if not pred_candidates:
            continue
        after = cv2.imread(str(pred_candidates[0]), cv2.IMREAD_COLOR)
        if clean is None or before is None or after is None:
            continue
        if after.shape != clean.shape:
            after = cv2.resize(after, (clean.shape[1], clean.shape[0]))
        if before.shape != clean.shape:
            before = cv2.resize(before, (clean.shape[1], clean.shape[0]))

        rec_metrics = {
            "sample_id": rec["sample_id"],
            "method": args.method,
            "psnr": psnr(clean, after),
            "ssim": ssim_gray(clean, after),
            "edge_keep": edge_consistency(before, after),
        }
        if args.binary:
            rec_metrics.update(binary_consistency(clean, before, after))
        metrics.append(rec_metrics)

    summary = {}
    if metrics:
        keys = [k for k in metrics[0] if k not in {"sample_id", "method"}]
        summary = {"n": len(metrics), "method": args.method}
        for k in keys:
            summary[k] = sum(m[k] for m in metrics) / len(metrics)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "samples": metrics}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
