import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def by_id(path):
    return {r["sample_id"]: r for r in read_jsonl(path)}


def ci(vals, seed=20260813, n_boot=8000):
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=np.float64)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.mean(arr)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    out = Path("outputs/mmdir_augmented_selector_20260813/rp0p55_hp0p15_sw8")
    mmdir = by_id(out / "test_mmdir_official.jsonl")
    hist = by_id(out / "test_histrestore_mmdir_prior_selector.jsonl")
    ids = sorted(set(mmdir) & set(hist))
    metrics = [
        ("psnr", "higher"),
        ("ssim", "higher"),
        ("risk", "lower"),
    ]
    rows = []
    for metric, direction in metrics:
        deltas = [hist[sid][metric] - mmdir[sid][metric] for sid in ids]
        mean, lo, hi = ci(deltas)
        win = float(np.mean([d > 0 for d in deltas])) if direction == "higher" else float(np.mean([d < 0 for d in deltas]))
        rows.append(
            {
                "comparison": "HistRestore+MMDIR vs MMDIR official",
                "metric": metric,
                "n": len(ids),
                "mean_delta": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "win_rate": win,
            }
        )
    with (out / "mmdir_augmented_bootstrap.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "mmdir_augmented_bootstrap.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    psnr = rows[0]
    ax.bar([0], [psnr["mean_delta"]], color="#16a34a")
    ax.errorbar([0], [psnr["mean_delta"]], yerr=[[psnr["mean_delta"] - psnr["ci95_low"]], [psnr["ci95_high"] - psnr["mean_delta"]]], fmt="none", ecolor="#111827", capsize=4)
    ax.axhline(0, color="#111827", lw=0.8)
    ax.set_xticks([0])
    ax.set_xticklabels(["HistRestore+MMDIR\nvs MMDIR"])
    ax.set_ylabel("PSNR delta / dB")
    ax.set_title("MixedDoc Test Bootstrap 95% CI")
    fig.tight_layout()
    fig.savefig(out / "mmdir_augmented_bootstrap.png", dpi=180)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
