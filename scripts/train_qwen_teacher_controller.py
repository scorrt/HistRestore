import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from train_mixeddoc_noref_selector import add_features, load_rows


TARGETS = {
    "use_docres": "docres_deshadow",
    "blend_docres": "docres_input_blend_0.90",
    "use_classical": "classical_shadow_0.40",
    "preserve_input": "input",
    "manual_review": "classical_shadow_0.40",
}


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


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_reviews(paths):
    labels = {}
    for path in paths:
        for row in read_jsonl(path):
            sid = row["sample_id"]
            decision = row["qwen"]["decision"]
            labels[sid] = TARGETS.get(decision, "classical_shadow_0.40")
    return labels


def feature_for(cands):
    add_features(cands)
    base = cands["classical_shadow_0.40"]
    doc = cands["docres_deshadow"]
    app = cands.get("docres_appearance", base)
    region = cands.get("region_controller", base)
    vals = []
    for cand in [doc, app, region, base]:
        for name in FEATURES:
            vals.append(float(cand.get(name, 0.0)))
    vals += [
        doc["content_risk"] - base["content_risk"],
        app["content_risk"] - base["content_risk"],
        region["content_risk"] - base["content_risk"],
        doc["edge_keep"] - base["edge_keep"],
        app["edge_keep"] - base["edge_keep"],
        region["edge_keep"] - base["edge_keep"],
    ]
    return vals


def evaluate(samples, model, class_names, allowed):
    rows = []
    for sid, cands in samples.items():
        if "classical_shadow_0.40" not in cands or "docres_deshadow" not in cands:
            continue
        x = np.asarray([feature_for(cands)])
        probs = model.predict_proba(x)[0]
        ranked = sorted(zip(probs, class_names), reverse=True)
        chosen = None
        for _, name in ranked:
            if name in cands and name in allowed:
                chosen = cands[name]
                break
        if chosen is None:
            chosen = cands["classical_shadow_0.40"]
        base = cands["classical_shadow_0.40"]
        oracle = max([r for m, r in cands.items() if m in allowed], key=lambda r: r["psnr"])
        rows.append(
            {
                "sample_id": sid,
                "candidate": chosen["method"],
                "psnr": chosen["psnr"],
                "ssim": chosen["ssim"],
                "risk": chosen["content_risk"],
                "gain_vs_classical": chosen["psnr"] - base["psnr"],
                "oracle_psnr": oracle["psnr"],
                "harm": float(chosen["psnr"] < base["psnr"]),
            }
        )
    return {
        "n": len(rows),
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "risk": float(np.mean([r["risk"] for r in rows])),
        "gain_vs_classical": float(np.mean([r["gain_vs_classical"] for r in rows])),
        "harm_rate": float(np.mean([r["harm"] for r in rows])),
        "selected_counts": dict(Counter(r["candidate"] for r in rows)),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--qwen-reviews", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    samples = load_rows(args.candidate_metrics)
    labels = load_reviews(args.qwen_reviews)
    train_sids = [sid for sid in labels if sid in samples and labels[sid] in samples[sid]]
    x, y = [], []
    for sid in train_sids:
        x.append(feature_for(samples[sid]))
        y.append(labels[sid])
    class_names = sorted(set(y))
    model = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.035, random_state=37)
    model.fit(np.asarray(x), np.asarray(y))
    allowed = [
        "input",
        "classical_shadow_0.40",
        "docres_deshadow",
        "docres_input_blend_0.90",
        "docres_input_blend_0.80",
        "region_controller",
    ]
    full = evaluate(samples, model, class_names, allowed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "teacher_samples": len(train_sids),
        "teacher_label_counts": dict(Counter(y)),
        "model": "GradientBoostingClassifier distilled from Qwen review decisions",
        "allowed_candidates": allowed,
        "full": {k: v for k, v in full.items() if k != "rows"},
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "per_sample.jsonl").open("w", encoding="utf-8") as f:
        for r in full["rows"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    methods = list(full["selected_counts"])
    vals = [full["selected_counts"][m] for m in methods]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(methods, vals, color="#2563eb")
    ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
    ax.set_title("Qwen-Teacher Controller Selected Candidates")
    fig.tight_layout()
    fig.savefig(out / "qwen_teacher_controller_counts.png", dpi=180)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
