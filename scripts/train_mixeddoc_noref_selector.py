import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FEATURES = [
    "edge_keep",
    "edge_jaccard",
    "foreground_shift",
    "mean_shift",
    "contrast_shift",
    "contrast_after",
    "sharp_after",
    "content_risk",
    "risk_over_base",
]


def stable_hash(text):
    h = 2166136261
    for ch in text.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def load_rows(path):
    by_sample = defaultdict(dict)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sid = r["sample_id"]
        by_sample[sid][r["method"]] = r
    return by_sample


def add_features(cands):
    base = cands.get("classical_shadow_0.40")
    for r in cands.values():
        r["risk_over_base"] = max(0.0, r["content_risk"] - base["content_risk"]) if base else 0.0
        r["sharp_log"] = min(np.log1p(max(r["sharp_after"], 0.0)) / 8.0, 1.5)
        r["contrast_norm"] = min(r["contrast_after"] / 64.0, 1.5)


def score_candidate(r, weights, priors):
    return (
        priors.get(r["method"], 0.0)
        + weights["edge_keep"] * r["edge_keep"]
        + weights["edge_jaccard"] * r["edge_jaccard"]
        + weights["contrast_after"] * r["contrast_norm"]
        + weights["sharp_after"] * r["sharp_log"]
        - weights["content_risk"] * r["content_risk"]
        - weights["risk_over_base"] * r["risk_over_base"]
        - weights["foreground_shift"] * min(r["foreground_shift"] * 2.0, 1.0)
        - weights["mean_shift"] * min(r["mean_shift"] * 4.0, 1.0)
        - weights["contrast_shift"] * min(r["contrast_shift"] * 3.0, 1.0)
    )


def select(cands, weights, priors, margin):
    best = max(cands.values(), key=lambda r: score_candidate(r, weights, priors))
    base = cands.get("classical_shadow_0.40")
    if base is not None and score_candidate(best, weights, priors) < score_candidate(base, weights, priors) + margin:
        return base
    return best


def evaluate(samples, weights, priors, margin, risk_penalty=0.0):
    selected = []
    for sid, cands in samples.items():
        add_features(cands)
        if "classical_shadow_0.40" not in cands:
            continue
        chosen = select(cands, weights, priors, margin)
        base = cands["classical_shadow_0.40"]
        oracle = max(cands.values(), key=lambda r: r["psnr"] - risk_penalty * max(0.0, r["content_risk"] - base["content_risk"]))
        selected.append((sid, chosen, base, oracle))
    if not selected:
        return None
    return {
        "n": len(selected),
        "psnr": float(np.mean([c[1]["psnr"] for c in selected])),
        "ssim": float(np.mean([c[1]["ssim"] for c in selected])),
        "risk": float(np.mean([c[1]["content_risk"] for c in selected])),
        "gain_vs_classical": float(np.mean([c[1]["psnr"] - c[2]["psnr"] for c in selected])),
        "oracle_gain_vs_classical": float(np.mean([c[3]["psnr"] - c[2]["psnr"] for c in selected])),
        "harm_rate": float(np.mean([1.0 if c[1]["psnr"] < c[2]["psnr"] else 0.0 for c in selected])),
        "selected_counts": Counter(c[1]["method"] for c in selected),
        "selected": selected,
    }


def random_weights(rng):
    weights = {
        "edge_keep": rng.uniform(0.10, 0.85),
        "edge_jaccard": rng.uniform(0.00, 0.45),
        "contrast_after": rng.uniform(0.00, 0.35),
        "sharp_after": rng.uniform(0.00, 0.30),
        "content_risk": rng.uniform(0.55, 1.70),
        "risk_over_base": rng.uniform(0.20, 1.35),
        "foreground_shift": rng.uniform(0.10, 0.80),
        "mean_shift": rng.uniform(0.00, 0.35),
        "contrast_shift": rng.uniform(0.00, 0.55),
    }
    priors = {
        "input": rng.uniform(-0.08, 0.10),
        "classical_shadow_0.30": rng.uniform(0.12, 0.35),
        "classical_shadow_0.40": rng.uniform(0.15, 0.40),
        "classical_shadow_0.50": rng.uniform(0.12, 0.35),
        "docres_deshadow": rng.uniform(0.55, 1.25),
        "docres_input_blend_0.90": rng.uniform(0.50, 1.10),
        "docres_input_blend_0.80": rng.uniform(0.42, 0.95),
        "docres_input_blend_0.70": rng.uniform(0.35, 0.85),
        "docres_classical040_blend_0.90": rng.uniform(0.45, 1.00),
        "docres_classical040_blend_0.80": rng.uniform(0.35, 0.90),
        "docres_appearance": rng.uniform(0.30, 0.85),
        "appearance_input_blend_0.90": rng.uniform(0.25, 0.75),
        "appearance_input_blend_0.80": rng.uniform(0.20, 0.70),
    }
    return weights, priors, rng.uniform(0.00, 0.12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trials", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--risk-penalty", type=float, default=0.85)
    args = parser.parse_args()

    by_sample = load_rows(args.candidate_metrics)
    by_sample = {sid: c for sid, c in by_sample.items() if "classical_shadow_0.40" in c and "docres_deshadow" in c}
    train = {sid: c for sid, c in by_sample.items() if stable_hash(sid) % 5 < 3}
    val = {sid: c for sid, c in by_sample.items() if stable_hash(sid) % 5 == 3}
    test = {sid: c for sid, c in by_sample.items() if stable_hash(sid) % 5 == 4}
    rng = np.random.default_rng(args.seed)

    best = None
    history = []
    for i in range(args.trials):
        weights, priors, margin = random_weights(rng)
        tr = evaluate(train, weights, priors, margin, args.risk_penalty)
        va = evaluate(val, weights, priors, margin, args.risk_penalty)
        if not tr or not va:
            continue
        # Utility favors PSNR gain, but discourages high content-risk movement and high harm rate.
        util = va["gain_vs_classical"] - args.risk_penalty * max(0.0, va["risk"] - 0.16) - 0.35 * va["harm_rate"]
        rec = {"trial": i, "utility": util, "train": tr, "val": va, "weights": weights, "priors": priors, "margin": margin}
        history.append({k: v for k, v in rec.items() if k not in {"train", "val"}} | {"val_gain": va["gain_vs_classical"], "val_risk": va["risk"], "val_harm": va["harm_rate"]})
        if best is None or util > best["utility"]:
            best = rec

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    te = evaluate(test, best["weights"], best["priors"], best["margin"], args.risk_penalty)
    full = evaluate(by_sample, best["weights"], best["priors"], best["margin"], args.risk_penalty)
    summary = {
        "sample_counts": {"all": len(by_sample), "train": len(train), "val": len(val), "test": len(test)},
        "risk_penalty": args.risk_penalty,
        "best_trial": best["trial"],
        "weights": best["weights"],
        "priors": best["priors"],
        "margin": best["margin"],
        "train": {k: v for k, v in best["train"].items() if k != "selected"},
        "val": {k: v for k, v in best["val"].items() if k != "selected"},
        "test": {k: v for k, v in te.items() if k != "selected"},
        "full": {k: v for k, v in full.items() if k != "selected"},
    }
    for split in ["train", "val", "test", "full"]:
        summary[split]["selected_counts"] = dict(summary[split]["selected_counts"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "per_sample_test.jsonl").open("w", encoding="utf-8") as f:
        for sid, chosen, base, oracle in te["selected"]:
            f.write(
                json.dumps(
                    {
                        "sample_id": sid,
                        "selected_candidate": chosen["method"],
                        "oracle_candidate": oracle["method"],
                        "selected_minus_classical_psnr": chosen["psnr"] - base["psnr"],
                        "oracle_minus_classical_psnr": oracle["psnr"] - base["psnr"],
                        "selected_risk": chosen["content_risk"],
                        "classical_risk": base["content_risk"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    history = sorted(history, key=lambda x: x["utility"], reverse=True)[:100]
    with (out / "top_trials.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["trial", "utility", "val_gain", "val_risk", "val_harm", "margin"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for h in history:
            writer.writerow({k: h[k] for k in fields})

    methods = ["train", "val", "test", "full"]
    gains = [summary[m]["gain_vs_classical"] for m in methods]
    risks = [summary[m]["risk"] for m in methods]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(methods))
    ax1.bar(x - 0.18, gains, width=0.36, color="#2563eb", label="PSNR gain vs classical")
    ax1.set_ylabel("PSNR gain (dB)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, risks, width=0.36, color="#f97316", label="content risk")
    ax2.set_ylabel("Content risk")
    ax1.set_title("MixedDoc no-reference selector tuning")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "split_gain_risk.png", dpi=180)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
