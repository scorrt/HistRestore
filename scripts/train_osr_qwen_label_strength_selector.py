import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from train_osr_strength_selector_split import METHODS, cand_feature, read_rows, sample_feature, summarize, baseline


def load_qwen_labels(path):
    labels = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        choice = row.get("choice") or row.get("qwen", {}).get("choice")
        if choice in METHODS:
            labels[row["sample_id"]] = choice
    return labels


def train_qwen_model(samples, labels, train_ids):
    sids = [sid for sid in train_ids if sid in labels and sid in samples]
    x = [sample_feature(samples[sid]) for sid in sids]
    y = [labels[sid] for sid in sids]
    model = GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.04, min_samples_leaf=3, random_state=131)
    model.fit(np.asarray(x), np.asarray(y))
    return model, list(model.classes_), sids, Counter(y)


def qprior_probs(model, classes, cands):
    probs = model.predict_proba(np.asarray([sample_feature(cands)]))[0]
    out = {m: 0.0 for m in METHODS}
    for cls, prob in zip(classes, probs):
        out[cls] = float(prob)
    out["preserve_or_light"] = out.get("input", 0.0) + out.get("shadow_0.30", 0.0)
    return out


def train_selector(samples, train_ids, use_prior, qmodel, qclasses, ssim_weight, risk_penalty, harm_penalty):
    x, y = [], []
    for sid in train_ids:
        cands = samples[sid]
        base = cands["shadow_0.40"]
        qprior = qprior_probs(qmodel, qclasses, cands) if use_prior else None
        for m in METHODS:
            row = cands[m]
            x.append(cand_feature(row, base, qprior))
            y.append(row["psnr"] + ssim_weight * row["ssim"] - risk_penalty * row["content_risk"] - harm_penalty * max(0.0, base["psnr"] - row["psnr"]))
    model = GradientBoostingRegressor(n_estimators=220, max_depth=3, learning_rate=0.025, min_samples_leaf=4, random_state=151 if use_prior else 149)
    model.fit(np.asarray(x), np.asarray(y))
    return model


def evaluate(samples, ids, model, use_prior, qmodel, qclasses):
    rows = []
    for sid in ids:
        cands = samples[sid]
        base = cands["shadow_0.40"]
        qprior = qprior_probs(qmodel, qclasses, cands) if use_prior else None
        scored = []
        for m in METHODS:
            row = cands[m]
            scored.append((float(model.predict(np.asarray([cand_feature(row, base, qprior)]))[0]), m, row))
        _, cand, row = max(scored, key=lambda z: z[0])
        oracle = max((cands[m] for m in METHODS), key=lambda r: (r["psnr"], r["ssim"]))
        rows.append(
            {
                "sample_id": sid,
                "candidate": cand,
                "psnr": row["psnr"],
                "ssim": row["ssim"],
                "risk": row["content_risk"],
                "alpha": row["alpha"],
                "gain_vs_shadow040": row["psnr"] - base["psnr"],
                "harm": float(row["psnr"] < base["psnr"]),
                "gap_to_oracle": oracle["psnr"] - row["psnr"],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--qwen-labels", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ssim-weight", type=float, default=3.0)
    parser.add_argument("--risk-penalty", type=float, default=0.6)
    parser.add_argument("--harm-penalty", type=float, default=0.4)
    args = parser.parse_args()
    samples = read_rows(args.candidate_metrics)
    splits = defaultdict(list)
    for sid, cands in samples.items():
        splits[cands["input"]["split"]].append(sid)
    train_ids = splits["train"]
    labels = load_qwen_labels(args.qwen_labels)
    qmodel, qclasses, teacher_sids, label_counts = train_qwen_model(samples, labels, train_ids)
    no_model = train_selector(samples, train_ids, False, qmodel, qclasses, args.ssim_weight, args.risk_penalty, args.harm_penalty)
    q_model = train_selector(samples, train_ids, True, qmodel, qclasses, args.ssim_weight, args.risk_penalty, args.harm_penalty)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": "Qwen candidate-comparison labels on 80 train hard cases; distilled and evaluated on stable val split",
        "teacher_samples": len(teacher_sids),
        "teacher_label_counts": dict(label_counts),
        "qclasses": qclasses,
        "params": {"ssim_weight": args.ssim_weight, "risk_penalty": args.risk_penalty, "harm_penalty": args.harm_penalty},
        "splits": {},
    }
    for split, ids in splits.items():
        rows = {
            "shadow_0.30": baseline(samples, ids, "shadow_0.30"),
            "shadow_0.40": baseline(samples, ids, "shadow_0.40"),
            "no_qwen_selector": evaluate(samples, ids, no_model, False, qmodel, qclasses),
            "qwen_label_prior_selector": evaluate(samples, ids, q_model, True, qmodel, qclasses),
        }
        summary["splits"][split] = {name: summarize(r) for name, r in rows.items()}
        summary["splits"][split]["delta_qwen_vs_no_qwen"] = {
            "psnr": summary["splits"][split]["qwen_label_prior_selector"]["psnr"] - summary["splits"][split]["no_qwen_selector"]["psnr"],
            "ssim": summary["splits"][split]["qwen_label_prior_selector"]["ssim"] - summary["splits"][split]["no_qwen_selector"]["ssim"],
            "risk": summary["splits"][split]["qwen_label_prior_selector"]["risk"] - summary["splits"][split]["no_qwen_selector"]["risk"],
            "harm_rate": summary["splits"][split]["qwen_label_prior_selector"]["harm_rate"] - summary["splits"][split]["no_qwen_selector"]["harm_rate"],
        }
        for name, r in rows.items():
            with (out / f"{split}_{name}.jsonl").open("w", encoding="utf-8") as f:
                for row in r:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = ["shadow_0.30", "shadow_0.40", "no_qwen_selector", "qwen_label_prior_selector"]
    labels_plot = ["0.30", "0.40", "No-Qwen", "Qwen labels"]
    vals = [summary["splits"]["val"][k]["psnr"] for k in keys]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(labels_plot, vals, color=["#64748b", "#64748b", "#2563eb", "#16a34a"])
    ax.set_ylabel("Val PSNR")
    ax.set_title("OSR Real Qwen Candidate-Comparison Prior")
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "osr_real_qwen_prior_val.png", dpi=180)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
