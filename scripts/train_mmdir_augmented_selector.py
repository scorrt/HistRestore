import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

sys.path.append(str(Path(__file__).resolve().parent))
from evaluate_mmdir_official_mixeddoc import read_jsonl, summarize_metric_rows, write_csv  # noqa: E402
from train_mixeddoc_noref_selector import stable_hash  # noqa: E402


METHODS = [
    "input_degraded",
    "classical_shadow_0.40",
    "docres_deshadow",
    "docres_input_blend_0.90",
    "docres_input_blend_0.80",
    "docres_classical040_blend_0.90",
    "region_controller",
    "mmdir_official",
]


def split_name(sample_id):
    bucket = stable_hash(sample_id) % 10
    if bucket < 6:
        return "train"
    if bucket < 8:
        return "val"
    return "test"


def group_rows(*paths):
    grouped = defaultdict(dict)
    for path in paths:
        for r in read_jsonl(path):
            method = r["method"]
            if method in METHODS:
                grouped[r["sample_id"]][method] = r
    return grouped


def normalized_method_features(name):
    families = {
        "input_degraded": [1, 0, 0, 0],
        "classical_shadow_0.40": [0, 1, 0, 0],
        "docres_deshadow": [0, 0, 1, 0],
        "docres_input_blend_0.90": [0, 0, 1, 0],
        "docres_input_blend_0.80": [0, 0, 1, 0],
        "docres_classical040_blend_0.90": [0, 0, 1, 0],
        "region_controller": [0, 0, 1, 0],
        "mmdir_official": [0, 0, 0, 1],
    }
    return families.get(name, [0, 0, 0, 0])


def feature(row, base, mmdir, method):
    risk = row.get("content_risk", row.get("risk", 0.0))
    base_risk = base.get("content_risk", base.get("risk", 0.0))
    mmdir_risk = mmdir.get("content_risk", mmdir.get("risk", 0.0))
    vals = [
        row.get("edge_jaccard", 0.0),
        row.get("edge_keep", 0.0),
        row.get("foreground_shift", 0.0),
        row.get("mean_shift", 0.0),
        row.get("contrast_shift", 0.0),
        min(row.get("contrast_after", 0.0) / 80.0, 2.0),
        min(np.log1p(max(row.get("sharp_after", 0.0), 0.0)) / 8.0, 2.0),
        risk,
        risk - base_risk,
        risk - mmdir_risk,
        max(0.0, risk - 0.30),
        max(0.0, risk - 0.45),
        row.get("edge_keep", 0.0) - base.get("edge_keep", 0.0),
        row.get("foreground_shift", 0.0) - base.get("foreground_shift", 0.0),
    ]
    vals += [1.0 if method == m else 0.0 for m in METHODS]
    vals += normalized_method_features(method)
    return vals


def utility(row, base, ssim_weight, risk_penalty, harm_penalty):
    risk = row.get("content_risk", row.get("risk", 0.0))
    base_risk = base.get("content_risk", base.get("risk", 0.0))
    return (
        row["psnr"]
        + ssim_weight * row["ssim"]
        - risk_penalty * risk
        - 0.65 * risk_penalty * max(0.0, risk - base_risk)
        - harm_penalty * max(0.0, base["psnr"] - row["psnr"])
    )


def train_regressor(samples, ids, ssim_weight, risk_penalty, harm_penalty):
    x, y = [], []
    for sid in ids:
        c = samples[sid]
        if not all(m in c for m in ["classical_shadow_0.40", "mmdir_official"]):
            continue
        base = c["classical_shadow_0.40"]
        mmdir = c["mmdir_official"]
        for method in METHODS:
            if method in c:
                x.append(feature(c[method], base, mmdir, method))
                y.append(utility(c[method], base, ssim_weight, risk_penalty, harm_penalty))
    model = GradientBoostingRegressor(
        n_estimators=260,
        max_depth=3,
        learning_rate=0.025,
        min_samples_leaf=6,
        random_state=202613,
    )
    model.fit(np.asarray(x), np.asarray(y))
    return model


def train_qwen_like_prior(samples, ids):
    # Offline teacher: MMDIR is a strong restoration candidate, but we train a
    # candidate-family prior only from evidence and oracle labels on train split.
    x, y = [], []
    for sid in ids:
        c = samples[sid]
        if not all(m in c for m in ["classical_shadow_0.40", "mmdir_official"]):
            continue
        base = c["classical_shadow_0.40"]
        mmdir = c["mmdir_official"]
        oracle = max((m for m in METHODS if m in c), key=lambda m: (c[m]["psnr"], c[m]["ssim"]))
        sample_feat = []
        for method in METHODS:
            if method in c:
                sample_feat += feature(c[method], base, mmdir, method)[:14]
            else:
                sample_feat += [0.0] * 14
        x.append(sample_feat)
        y.append(oracle)
    model = GradientBoostingClassifier(n_estimators=180, max_depth=3, learning_rate=0.03, min_samples_leaf=5, random_state=202614)
    model.fit(np.asarray(x), np.asarray(y))
    return model, list(model.classes_)


def qprior_features(model, classes, sample):
    base = sample["classical_shadow_0.40"]
    mmdir = sample["mmdir_official"]
    feat = []
    for method in METHODS:
        if method in sample:
            feat += feature(sample[method], base, mmdir, method)[:14]
        else:
            feat += [0.0] * 14
    probs = model.predict_proba(np.asarray([feat]))[0]
    out = {m: 0.0 for m in METHODS}
    for cls, p in zip(classes, probs):
        out[cls] = float(p)
    return out


def train_regressor_with_prior(samples, ids, prior_model, prior_classes, ssim_weight, risk_penalty, harm_penalty):
    x, y = [], []
    for sid in ids:
        c = samples[sid]
        if not all(m in c for m in ["classical_shadow_0.40", "mmdir_official"]):
            continue
        base = c["classical_shadow_0.40"]
        mmdir = c["mmdir_official"]
        qprior = qprior_features(prior_model, prior_classes, c)
        for method in METHODS:
            if method in c:
                x.append(feature(c[method], base, mmdir, method) + [qprior[m] for m in METHODS] + [qprior[method], qprior["mmdir_official"] * float(method == "mmdir_official")])
                y.append(utility(c[method], base, ssim_weight, risk_penalty, harm_penalty))
    model = GradientBoostingRegressor(
        n_estimators=280,
        max_depth=3,
        learning_rate=0.025,
        min_samples_leaf=6,
        random_state=202615,
    )
    model.fit(np.asarray(x), np.asarray(y))
    return model


def eval_model(samples, ids, model, prior_model=None, prior_classes=None):
    rows = []
    for sid in ids:
        c = samples[sid]
        if not all(m in c for m in ["classical_shadow_0.40", "mmdir_official"]):
            continue
        base = c["classical_shadow_0.40"]
        mmdir = c["mmdir_official"]
        qprior = qprior_features(prior_model, prior_classes, c) if prior_model else None
        scored = []
        for method in METHODS:
            if method not in c:
                continue
            feat = feature(c[method], base, mmdir, method)
            if qprior is not None:
                feat = feat + [qprior[m] for m in METHODS] + [qprior[method], qprior["mmdir_official"] * float(method == "mmdir_official")]
            pred = float(model.predict(np.asarray([feat]))[0])
            scored.append((pred, method, c[method]))
        _, chosen_name, chosen = max(scored, key=lambda z: z[0])
        oracle_name = max((m for m in METHODS if m in c), key=lambda m: (c[m]["psnr"], c[m]["ssim"]))
        oracle = c[oracle_name]
        rows.append(
            {
                "sample_id": sid,
                "candidate": chosen_name,
                "psnr": chosen["psnr"],
                "ssim": chosen["ssim"],
                "risk": chosen.get("content_risk", chosen.get("risk", 0.0)),
                "gain_vs_mmdir": chosen["psnr"] - c["mmdir_official"]["psnr"],
                "risk_vs_mmdir": chosen.get("content_risk", chosen.get("risk", 0.0)) - c["mmdir_official"].get("content_risk", c["mmdir_official"].get("risk", 0.0)),
                "gap_to_oracle": oracle["psnr"] - chosen["psnr"],
                "harm_vs_mmdir": float(chosen["psnr"] < c["mmdir_official"]["psnr"]),
                "oracle_candidate": oracle_name,
            }
        )
    return rows


def fixed_method(samples, ids, method):
    rows = []
    for sid in ids:
        if method not in samples[sid]:
            continue
        r = samples[sid][method]
        rows.append(
            {
                "sample_id": sid,
                "candidate": method,
                "psnr": r["psnr"],
                "ssim": r["ssim"],
                "risk": r.get("content_risk", r.get("risk", 0.0)),
                "gain_vs_mmdir": r["psnr"] - samples[sid]["mmdir_official"]["psnr"],
                "risk_vs_mmdir": r.get("content_risk", r.get("risk", 0.0)) - samples[sid]["mmdir_official"].get("content_risk", samples[sid]["mmdir_official"].get("risk", 0.0)),
                "gap_to_oracle": max(samples[sid][m]["psnr"] for m in METHODS if m in samples[sid]) - r["psnr"],
                "harm_vs_mmdir": float(r["psnr"] < samples[sid]["mmdir_official"]["psnr"]),
            }
        )
    return rows


def summarize(rows):
    return {
        "n": len(rows),
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "risk": float(np.mean([r["risk"] for r in rows])),
        "gain_vs_mmdir": float(np.mean([r["gain_vs_mmdir"] for r in rows])),
        "risk_vs_mmdir": float(np.mean([r["risk_vs_mmdir"] for r in rows])),
        "gap_to_oracle": float(np.mean([r["gap_to_oracle"] for r in rows])),
        "harm_vs_mmdir": float(np.mean([r["harm_vs_mmdir"] for r in rows])),
        "selected_counts": dict(Counter(r["candidate"] for r in rows)),
    }


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmdir-rows", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ssim-weight", type=float, default=8.0)
    parser.add_argument("--risk-penalty", type=float, default=0.55)
    parser.add_argument("--harm-penalty", type=float, default=0.15)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    samples = group_rows(args.mmdir_rows, args.candidate_metrics)
    samples = {sid: c for sid, c in samples.items() if "mmdir_official" in c and "classical_shadow_0.40" in c}
    splits = defaultdict(list)
    for sid in sorted(samples):
        splits[split_name(sid)].append(sid)

    base_model = train_regressor(samples, splits["train"], args.ssim_weight, args.risk_penalty, args.harm_penalty)
    prior_model, prior_classes = train_qwen_like_prior(samples, splits["train"])
    prior_selector = train_regressor_with_prior(samples, splits["train"], prior_model, prior_classes, args.ssim_weight, args.risk_penalty, args.harm_penalty)

    summary = {
        "protocol": "MMDIR official predictions added as a strong candidate/backbone. Stable hash 60/20/20 split; selectors trained on train split only.",
        "sample_counts": {k: len(v) for k, v in splits.items()},
        "params": {"ssim_weight": args.ssim_weight, "risk_penalty": args.risk_penalty, "harm_penalty": args.harm_penalty},
        "prior_classes": prior_classes,
        "splits": {},
    }
    for split, ids in splits.items():
        rows = {
            "input_degraded": fixed_method(samples, ids, "input_degraded"),
            "classical_shadow_0.40": fixed_method(samples, ids, "classical_shadow_0.40"),
            "docres_deshadow": fixed_method(samples, ids, "docres_deshadow"),
            "mmdir_official": fixed_method(samples, ids, "mmdir_official"),
            "histrestore_mmdir_selector": eval_model(samples, ids, base_model),
            "histrestore_mmdir_prior_selector": eval_model(samples, ids, prior_selector, prior_model, prior_classes),
            "oracle_mmdir_pool": [],
        }
        for sid in ids:
            best = max((m for m in METHODS if m in samples[sid]), key=lambda m: (samples[sid][m]["psnr"], samples[sid][m]["ssim"]))
            rows["oracle_mmdir_pool"].append({**fixed_method(samples, [sid], best)[0], "candidate": best})
        summary["splits"][split] = {name: summarize(r) for name, r in rows.items()}
        for name, r in rows.items():
            write_jsonl(out / f"{split}_{name}.jsonl", r)

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    flat = []
    for split, rec in summary["splits"].items():
        for method, vals in rec.items():
            row = {"split": split, "method": method}
            row.update({k: v for k, v in vals.items() if k != "selected_counts"})
            flat.append(row)
    write_csv(out / "split_summary.csv", flat)

    methods = ["input_degraded", "docres_deshadow", "mmdir_official", "histrestore_mmdir_selector", "histrestore_mmdir_prior_selector", "oracle_mmdir_pool"]
    test = summary["splits"]["test"]
    fig, ax1 = plt.subplots(figsize=(10.2, 4.8))
    x = np.arange(len(methods))
    ax1.bar(x - 0.18, [test[m]["psnr"] for m in methods], width=0.36, color="#2563eb", label="PSNR")
    ax1.set_ylabel("Test PSNR / dB")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, [test[m]["risk"] for m in methods], width=0.36, color="#f97316", label="risk")
    ax2.set_ylabel("Content risk")
    ax1.set_title("MixedDoc Test: HistRestore With MMDIR Strong Candidate")
    fig.tight_layout()
    fig.savefig(out / "test_mmdir_augmented_summary.png", dpi=180)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
