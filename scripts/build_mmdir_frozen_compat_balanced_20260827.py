import csv
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "mmdir_frozen_compat_balanced_20260827"
OUT.mkdir(parents=True, exist_ok=True)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mmdir = load_module(ROOT / "work" / "histrestore_first_round" / "train_mmdir_augmented_selector.py", "mmdir_aug")


METHODS = mmdir.METHODS
PARAMS = {"ssim_weight": 8.0, "risk_penalty": 0.55, "harm_penalty": 0.15}
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001]


def write_csv(path, rows):
    rows = list(rows)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def family(method):
    if method in {"input", "input_degraded"}:
        return "input"
    if method.startswith("classical") or method.startswith("shadow"):
        return "classical"
    if method == "mmdir_official":
        return "mmdir"
    if "region" in method:
        return "region"
    if "blend" in method:
        return "blend"
    if method.startswith("docres"):
        return "docres"
    return "other"


def strength(method):
    if family(method) == "input":
        return 0.0
    if family(method) == "classical":
        return 0.4
    if method == "docres_input_blend_0.80":
        return 0.8
    if method in {"docres_input_blend_0.90", "docres_classical040_blend_0.90"}:
        return 0.9
    if family(method) in {"docres", "region"}:
        return 0.9
    if family(method) == "mmdir":
        return 1.0
    return 0.5


def entropy_conf(qprior):
    vals = np.asarray([float(qprior.get(m, 0.0)) for m in METHODS], dtype=float)
    vals = vals / max(vals.sum(), 1e-12)
    ent = float(-(vals * np.log(vals + 1e-12)).sum())
    conf = 1.0 - ent / max(math.log(len(METHODS)), 1e-12)
    return ent, conf


def compat_balanced_features(method, row, qprior):
    """Frozen candidate-aware compatibility features.

    This is transferred as a generic prior-candidate compatibility family:
    no sample id, no split id, and no new protocol-specific tuning.
    """
    risk = float(row.get("content_risk", row.get("risk", 0.0)))
    edge = float(row.get("edge_keep", 0.0))
    cur = float(qprior.get(method, 0.0))
    fam = family(method)
    fam_sum = float(sum(qprior.get(m, 0.0) for m in METHODS if family(m) == fam))
    top = max(METHODS, key=lambda m: qprior.get(m, 0.0))
    top_fam = family(top)
    ent, conf = entropy_conf(qprior)
    target_strength = float(sum(qprior.get(m, 0.0) * strength(m) for m in METHODS))
    mild_prior = float(sum(qprior.get(m, 0.0) for m in METHODS if strength(m) <= 0.4))
    strong_prior = float(sum(qprior.get(m, 0.0) for m in METHODS if strength(m) >= 0.85))
    c_strength = strength(method)
    return [
        float(method == top),                         # policy/top-candidate match
        float(fam == top_fam),                        # family compatibility
        abs(c_strength - target_strength),            # strength gap
        risk * c_strength,                            # risk-strength interaction
        mild_prior * float(c_strength <= 0.4) * risk, # conservative match under risk
        strong_prior * float(c_strength >= 0.85) * risk, # aggressive conflict under risk
        cur * max(0.0, 1.0 - risk),                   # candidate prior after risk
        fam_sum * max(0.0, 1.0 - risk),               # family prior after risk
        cur * edge,                                   # content-edge compatibility
        conf,
        ent,
    ]


def base_feature(row, base, mmdir_row, method):
    return mmdir.feature(row, base, mmdir_row, method)


def utility(row, base):
    return mmdir.utility(row, base, **PARAMS)


def make_xy(samples, ids, prior_model, prior_classes, mode):
    x, y, row_ids, cand_names = [], [], [], []
    for sid in ids:
        c = samples[sid]
        base = c["classical_shadow_0.40"]
        mmdir_row = c["mmdir_official"]
        qprior = mmdir.qprior_features(prior_model, prior_classes, c)
        for method in METHODS:
            if method not in c:
                continue
            row = c[method]
            vals = base_feature(row, base, mmdir_row, method)
            if mode in {"raw_prior", "compat_balanced"}:
                vals += [qprior[m] for m in METHODS]
                vals += [qprior[method], qprior["mmdir_official"] * float(method == "mmdir_official")]
            if mode == "compat_balanced":
                vals += compat_balanced_features(method, row, qprior)
            x.append(vals)
            y.append(utility(row, base))
            row_ids.append(sid)
            cand_names.append(method)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), row_ids, cand_names


def train_model(samples, ids, prior_model, prior_classes, mode, estimator="GBDT", seed=202615):
    x, y, _, _ = make_xy(samples, ids, prior_model, prior_classes, mode)
    if estimator == "ExtraTrees":
        model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    else:
        model = GradientBoostingRegressor(
            n_estimators=280,
            max_depth=3,
            learning_rate=0.025,
            min_samples_leaf=6,
            random_state=seed,
        )
    model.fit(x, y)
    return model


def train_models(samples, ids, prior_model, prior_classes, mode, estimator):
    if estimator == "ExtraTrees":
        return [train_model(samples, ids, prior_model, prior_classes, mode, estimator=estimator, seed=s) for s in SEEDS]
    return [train_model(samples, ids, prior_model, prior_classes, mode, estimator=estimator, seed=202615)]


def oracle_row(samples, sid):
    best = max((m for m in METHODS if m in samples[sid]), key=lambda m: (samples[sid][m]["psnr"], samples[sid][m]["ssim"]))
    r = samples[sid][best]
    return {
        "sample_id": sid,
        "selected_candidate": best,
        "psnr": float(r["psnr"]),
        "ssim": float(r["ssim"]),
        "vccrp": float(r.get("content_risk", r.get("risk", 0.0))),
        "risk": float(r.get("content_risk", r.get("risk", 0.0))),
        "gain_vs_mmdir": float(r["psnr"] - samples[sid]["mmdir_official"]["psnr"]),
        "risk_vs_mmdir": float(r.get("content_risk", r.get("risk", 0.0)) - samples[sid]["mmdir_official"].get("content_risk", samples[sid]["mmdir_official"].get("risk", 0.0))),
        "gap_to_oracle": 0.0,
        "harm_vs_mmdir": 0.0,
    }


def fixed_row(samples, sid, method):
    r = samples[sid][method]
    oracle = oracle_row(samples, sid)
    risk = float(r.get("content_risk", r.get("risk", 0.0)))
    mmdir_risk = float(samples[sid]["mmdir_official"].get("content_risk", samples[sid]["mmdir_official"].get("risk", 0.0)))
    return {
        "sample_id": sid,
        "selected_candidate": method,
        "psnr": float(r["psnr"]),
        "ssim": float(r["ssim"]),
        "vccrp": risk,
        "risk": risk,
        "gain_vs_mmdir": float(r["psnr"] - samples[sid]["mmdir_official"]["psnr"]),
        "risk_vs_mmdir": float(risk - mmdir_risk),
        "gap_to_oracle": float(oracle["psnr"] - r["psnr"]),
        "harm_vs_mmdir": float(r["psnr"] < samples[sid]["mmdir_official"]["psnr"]),
    }


def select(samples, ids, models, prior_model, prior_classes, mode):
    x, _, row_ids, cand_names = make_xy(samples, ids, prior_model, prior_classes, mode)
    preds = np.vstack([model.predict(x) for model in models]).mean(axis=0)
    pred_map = defaultdict(list)
    for pred, sid, cand in zip(preds, row_ids, cand_names):
        pred_map[sid].append((float(pred), cand))
    rows = []
    for sid in ids:
        _, cand = max(pred_map[sid], key=lambda z: z[0])
        rows.append(fixed_row(samples, sid, cand) | {"method": mode})
    return rows


def summarize(rows, label, split):
    return {
        "split": split,
        "method": label,
        "n": len(rows),
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "vccrp": float(np.mean([r["vccrp"] for r in rows])),
        "harm_rate": float(np.mean([r["harm_vs_mmdir"] for r in rows])),
        "oracle_gap": float(np.mean([r["gap_to_oracle"] for r in rows])),
        "leave_mmdir_pages": int(sum(r["selected_candidate"] != "mmdir_official" for r in rows)),
        "leave_mmdir_rate": float(np.mean([r["selected_candidate"] != "mmdir_official" for r in rows])),
        "selected_counts": json.dumps(Counter(r["selected_candidate"] for r in rows).most_common(), ensure_ascii=False),
    }


def bootstrap(vals, n_boot=20000, seed=20260827):
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot = vals[idx].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def paired_stats(a_rows, b_rows, label):
    a = {r["sample_id"]: r for r in a_rows}
    b = {r["sample_id"]: r for r in b_rows}
    keys = sorted(set(a) & set(b))
    rows = []
    for metric, nice, higher in [("psnr", "PSNR", True), ("ssim", "SSIM", True), ("vccrp", "VCCRP", False), ("harm_vs_mmdir", "Harm", False), ("gap_to_oracle", "Oracle gap", False)]:
        diffs = np.asarray([float(a[k][metric]) - float(b[k][metric]) for k in keys])
        lo, hi = bootstrap(diffs)
        try:
            w_p = stats.wilcoxon(diffs, zero_method="wilcox").pvalue
        except ValueError:
            w_p = np.nan
        rows.append({
            "comparison": label,
            "metric": nice,
            "n": len(keys),
            "mean_delta": float(diffs.mean()),
            "ci95_low": lo,
            "ci95_high": hi,
            "paired_t_p": float(stats.ttest_rel([a[k][metric] for k in keys], [b[k][metric] for k in keys]).pvalue),
            "wilcoxon_p": float(w_p),
            "wins": int(np.sum(diffs > 1e-12 if higher else diffs < -1e-12)),
            "losses": int(np.sum(diffs < -1e-12 if higher else diffs > 1e-12)),
            "ties": int(np.sum(np.abs(diffs) <= 1e-12)),
            "win_rate": float(np.mean(diffs > 1e-12 if higher else diffs < -1e-12)),
        })
    return rows


def movement_stats(rows, label):
    moved = [r for r in rows if r["selected_candidate"] != "mmdir_official"]
    if not moved:
        return {
            "method": label,
            "leave_mmdir_pages": 0,
            "leave_mmdir_rate": 0.0,
            "moved_mean_gain_vs_mmdir": 0.0,
            "moved_mean_risk_vs_mmdir": 0.0,
            "moved_gain_pages": 0,
            "moved_loss_pages": 0,
            "moved_tie_pages": 0,
        }
    gains = np.asarray([r["gain_vs_mmdir"] for r in moved], dtype=float)
    risks = np.asarray([r["risk_vs_mmdir"] for r in moved], dtype=float)
    return {
        "method": label,
        "leave_mmdir_pages": len(moved),
        "leave_mmdir_rate": len(moved) / len(rows),
        "moved_mean_gain_vs_mmdir": float(gains.mean()),
        "moved_mean_risk_vs_mmdir": float(risks.mean()),
        "moved_gain_pages": int(np.sum(gains > 1e-12)),
        "moved_loss_pages": int(np.sum(gains < -1e-12)),
        "moved_tie_pages": int(np.sum(np.abs(gains) <= 1e-12)),
        "moved_candidate_counts": json.dumps(Counter(r["selected_candidate"] for r in moved).most_common(), ensure_ascii=False),
    }


def main():
    mmdir_rows = ROOT / "outputs" / "mmdir_same_protocol_all_20260813" / "mmdir_official_per_sample.jsonl"
    candidate_metrics = ROOT / "outputs" / "mixeddoc_candidate_cache_plus_region_20260812" / "candidate_metrics.jsonl"
    samples = mmdir.group_rows(mmdir_rows, candidate_metrics)
    samples = {sid: c for sid, c in samples.items() if "mmdir_official" in c and "classical_shadow_0.40" in c}
    splits = defaultdict(list)
    for sid in sorted(samples):
        splits[mmdir.split_name(sid)].append(sid)

    prior_model, prior_classes = mmdir.train_qwen_like_prior(samples, splits["train"])
    model_specs = [
        ("gbdt_raw_prior", "raw_prior", "GBDT"),
        ("gbdt_compat_balanced", "compat_balanced", "GBDT"),
        ("extratrees_raw_prior", "raw_prior", "ExtraTrees"),
        ("extratrees_compat_balanced", "compat_balanced", "ExtraTrees"),
    ]
    models = {
        label: train_models(samples, splits["train"], prior_model, prior_classes, mode, estimator)
        for label, mode, estimator in model_specs
    }

    all_summary, all_stats, all_move = [], [], []
    selected_by_split = {}
    for split, ids in splits.items():
        rows = {
            "mmdir_official": [fixed_row(samples, sid, "mmdir_official") | {"method": "mmdir_official"} for sid in ids],
            "oracle": [oracle_row(samples, sid) | {"method": "oracle"} for sid in ids],
        }
        for label, mode, _ in model_specs:
            rows[label] = select(samples, ids, models[label], prior_model, prior_classes, mode)
            for row in rows[label]:
                row["method"] = label
        selected_by_split[split] = rows
        for name, recs in rows.items():
            write_jsonl(OUT / f"{split}_{name}.jsonl", recs)
            all_summary.append(summarize(recs, name, split))
        if split == "test":
            all_stats += paired_stats(rows["gbdt_compat_balanced"], rows["gbdt_raw_prior"], "test GBDT compat_balanced vs raw_prior")
            all_stats += paired_stats(rows["extratrees_compat_balanced"], rows["extratrees_raw_prior"], "test ExtraTrees compat_balanced vs raw_prior")
            all_stats += paired_stats(rows["gbdt_compat_balanced"], rows["mmdir_official"], "test GBDT compat_balanced vs mmdir_official")
            all_stats += paired_stats(rows["extratrees_compat_balanced"], rows["mmdir_official"], "test ExtraTrees compat_balanced vs mmdir_official")
            all_stats += paired_stats(rows["gbdt_raw_prior"], rows["mmdir_official"], "test GBDT raw_prior vs mmdir_official")
            all_stats += paired_stats(rows["extratrees_raw_prior"], rows["mmdir_official"], "test ExtraTrees raw_prior vs mmdir_official")
            for label in ["gbdt_raw_prior", "gbdt_compat_balanced", "extratrees_raw_prior", "extratrees_compat_balanced"]:
                all_move.append(movement_stats(rows[label], label))

    write_csv(OUT / "summary.csv", all_summary)
    write_csv(OUT / "paired_stats.csv", all_stats)
    write_csv(OUT / "leave_mmdir_stats.csv", all_move)
    manifest = {
        "protocol": "MixedDoc Protocol B with MMDIR official candidate; stable_hash 60/20/20 split; no parameter tuning; raw prior vs frozen compat_balanced.",
        "params": PARAMS,
        "methods": METHODS,
        "splits": {k: len(v) for k, v in splits.items()},
        "prior_classes": prior_classes,
        "compat_features": [
            "top_candidate_match",
            "top_family_match",
            "strength_gap",
            "risk_candidate_strength",
            "mild_prior_under_risk",
            "strong_prior_risk_conflict",
            "candidate_prior_after_risk",
            "family_prior_after_risk",
            "candidate_prior_edge",
            "prior_confidence",
            "prior_entropy",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report = []
    report.append("# MixedDoc Protocol B: Frozen Compatibility-Balanced Prior")
    report.append("")
    report.append("This experiment inserts a frozen candidate-aware compatibility feature family into the existing 27.x MMDIR-augmented Protocol B. The split, candidate bank, utility parameters, prior classifier, and GBDT regressor settings are kept fixed.")
    report.append("")
    report.append("## Summary")
    report.append(pd.DataFrame(all_summary).drop(columns=["selected_counts"]).to_markdown(index=False))
    report.append("")
    report.append("## Paired Statistics")
    report.append(pd.DataFrame(all_stats).to_markdown(index=False))
    report.append("")
    report.append("## Leaving MMDIR")
    report.append(pd.DataFrame(all_move).to_markdown(index=False))
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "test_summary": [r for r in all_summary if r["split"] == "test"], "test_stats": all_stats, "leave_mmdir": all_move}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
