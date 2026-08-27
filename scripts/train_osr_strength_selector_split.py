import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


METHODS = ["input", "shadow_0.30", "shadow_0.40", "shadow_0.50", "shadow_0.60"]
FEATURES = [
    "alpha",
    "illum_reduction",
    "contrast_gain",
    "edge_keep",
    "foreground_shift",
    "contrast_shift",
    "mean_shift",
    "content_risk",
]


def read_rows(path):
    grouped = defaultdict(dict)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        grouped[row["sample_id"]][row["candidate"]] = row
    return {sid: c for sid, c in grouped.items() if all(m in c for m in METHODS)}


def cand_feature(row, base, qprior=None):
    vals = [float(row.get(k, 0.0)) for k in FEATURES]
    vals += [float(row.get(k, 0.0)) - float(base.get(k, 0.0)) for k in FEATURES[1:]]
    vals += [1.0 if row["candidate"] == m else 0.0 for m in METHODS]
    if qprior is not None:
        vals += [qprior.get(m, 0.0) for m in METHODS]
        vals += [qprior.get(row["candidate"], 0.0), qprior.get("preserve_or_light", 0.0) * row.get("alpha", 0.0)]
    return vals


def sample_feature(cands):
    base = cands["shadow_0.40"]
    light = cands["shadow_0.30"]
    strong = cands["shadow_0.60"]
    vals = []
    for row in [cands["input"], light, base, strong]:
        vals += [float(row.get(k, 0.0)) for k in FEATURES]
    vals += [
        light["content_risk"] - base["content_risk"],
        strong["content_risk"] - base["content_risk"],
        light["edge_keep"] - strong["edge_keep"],
        light["illum_reduction"] - strong["illum_reduction"],
        base["mean_shift"] - light["mean_shift"],
    ]
    return vals


def utility(row, base, ssim_weight, risk_penalty, harm_penalty):
    return (
        row["psnr"]
        + ssim_weight * row["ssim"]
        - risk_penalty * row["content_risk"]
        - harm_penalty * max(0.0, base["psnr"] - row["psnr"])
    )


def oracle_label(cands, ssim_weight, risk_penalty, harm_penalty):
    base = cands["shadow_0.40"]
    return max(METHODS, key=lambda m: utility(cands[m], base, ssim_weight, risk_penalty, harm_penalty))


def make_qprior_model(samples, train_ids, ssim_weight, risk_penalty, harm_penalty):
    x, y = [], []
    for sid in train_ids:
        cands = samples[sid]
        x.append(sample_feature(cands))
        y.append(oracle_label(cands, ssim_weight, risk_penalty, harm_penalty))
    model = GradientBoostingClassifier(n_estimators=160, max_depth=3, learning_rate=0.03, min_samples_leaf=4, random_state=73)
    model.fit(np.asarray(x), np.asarray(y))
    return model, list(model.classes_)


def qprior_probs(model, classes, cands):
    probs = model.predict_proba(np.asarray([sample_feature(cands)]))[0]
    out = {cls: float(p) for cls, p in zip(classes, probs)}
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
            y.append(utility(row, base, ssim_weight, risk_penalty, harm_penalty))
    model = GradientBoostingRegressor(n_estimators=220, max_depth=3, learning_rate=0.025, min_samples_leaf=4, random_state=91 if use_prior else 89)
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


def summarize(rows):
    return {
        "n": len(rows),
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "risk": float(np.mean([r["risk"] for r in rows])),
        "mean_alpha": float(np.mean([r["alpha"] for r in rows])),
        "gain_vs_shadow040": float(np.mean([r["gain_vs_shadow040"] for r in rows])),
        "harm_rate": float(np.mean([r["harm"] for r in rows])),
        "gap_to_oracle": float(np.mean([r["gap_to_oracle"] for r in rows])),
        "selected_counts": dict(Counter(r["candidate"] for r in rows)),
    }


def baseline(samples, ids, method):
    return [
        {
            "sample_id": sid,
            "candidate": method,
            "psnr": samples[sid][method]["psnr"],
            "ssim": samples[sid][method]["ssim"],
            "risk": samples[sid][method]["content_risk"],
            "alpha": samples[sid][method]["alpha"],
            "gain_vs_shadow040": samples[sid][method]["psnr"] - samples[sid]["shadow_0.40"]["psnr"],
            "harm": float(samples[sid][method]["psnr"] < samples[sid]["shadow_0.40"]["psnr"]),
            "gap_to_oracle": max(samples[sid][m]["psnr"] for m in METHODS) - samples[sid][method]["psnr"],
        }
        for sid in ids
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ssim-weight", type=float, default=3.0)
    parser.add_argument("--risk-penalty", type=float, default=1.0)
    parser.add_argument("--harm-penalty", type=float, default=0.25)
    args = parser.parse_args()
    samples = read_rows(args.candidate_metrics)
    splits = {"train": [], "val": []}
    for sid, cands in samples.items():
        splits[cands["input"]["split"]].append(sid)
    train_ids = splits["train"]
    qmodel, qclasses = make_qprior_model(samples, train_ids, args.ssim_weight, args.risk_penalty, args.harm_penalty)
    no_model = train_selector(samples, train_ids, False, qmodel, qclasses, args.ssim_weight, args.risk_penalty, args.harm_penalty)
    q_model = train_selector(samples, train_ids, True, qmodel, qclasses, args.ssim_weight, args.risk_penalty, args.harm_penalty)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": "stable_hash train/val; prior model trained on train only; sample ids and dataset identity are not features",
        "params": {"ssim_weight": args.ssim_weight, "risk_penalty": args.risk_penalty, "harm_penalty": args.harm_penalty},
        "qclasses": qclasses,
        "splits": {},
    }
    for split, ids in splits.items():
        rows = {
            "input": baseline(samples, ids, "input"),
            "shadow_0.30": baseline(samples, ids, "shadow_0.30"),
            "shadow_0.40": baseline(samples, ids, "shadow_0.40"),
            "no_prior_selector": evaluate(samples, ids, no_model, False, qmodel, qclasses),
            "qwen_like_prior_selector": evaluate(samples, ids, q_model, True, qmodel, qclasses),
        }
        oracle_rows = []
        for sid in ids:
            best = max((samples[sid][m] for m in METHODS), key=lambda r: (r["psnr"], r["ssim"]))
            base = samples[sid]["shadow_0.40"]
            oracle_rows.append(
                {
                    "sample_id": sid,
                    "candidate": best["candidate"],
                    "psnr": best["psnr"],
                    "ssim": best["ssim"],
                    "risk": best["content_risk"],
                    "alpha": best["alpha"],
                    "gain_vs_shadow040": best["psnr"] - base["psnr"],
                    "harm": float(best["psnr"] < base["psnr"]),
                    "gap_to_oracle": 0.0,
                }
            )
        rows["oracle_pool"] = oracle_rows
        for name, r in rows.items():
            with (out / f"{split}_{name}.jsonl").open("w", encoding="utf-8") as f:
                for row in r:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["splits"][split] = {name: summarize(r) for name, r in rows.items()}
        summary["splits"][split]["delta_qprior_vs_no_prior"] = {
            "psnr": summary["splits"][split]["qwen_like_prior_selector"]["psnr"] - summary["splits"][split]["no_prior_selector"]["psnr"],
            "ssim": summary["splits"][split]["qwen_like_prior_selector"]["ssim"] - summary["splits"][split]["no_prior_selector"]["ssim"],
            "risk": summary["splits"][split]["qwen_like_prior_selector"]["risk"] - summary["splits"][split]["no_prior_selector"]["risk"],
            "harm_rate": summary["splits"][split]["qwen_like_prior_selector"]["harm_rate"] - summary["splits"][split]["no_prior_selector"]["harm_rate"],
        }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    labels = ["input", "0.30", "0.40", "No prior", "Qwen-prior", "Oracle"]
    keys = ["input", "shadow_0.30", "shadow_0.40", "no_prior_selector", "qwen_like_prior_selector", "oracle_pool"]
    vals = [summary["splits"]["val"][k]["psnr"] for k in keys]
    risks = [summary["splits"]["val"][k]["risk"] for k in keys]
    x = np.arange(len(keys))
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.bar(x, vals, color=["#94a3b8", "#64748b", "#64748b", "#2563eb", "#16a34a", "#111827"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Val PSNR")
    ax2 = ax1.twinx()
    ax2.plot(x, risks, color="#f97316", marker="o")
    ax2.set_ylabel("Val risk")
    ax1.set_title("OSR Strength Selection With Prior Constraint")
    fig.tight_layout()
    fig.savefig(out / "osr_strength_prior_val.png", dpi=180)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
