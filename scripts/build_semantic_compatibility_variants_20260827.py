import csv
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "semantic_compatibility_variants_20260827"
OUT.mkdir(parents=True, exist_ok=True)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hist = load_module(ROOT / "work" / "histrestore_first_round" / "build_historical537_group_split_full_audit.py", "hist")
amb = load_module(ROOT / "work" / "histrestore_first_round" / "build_ambiguity_gated_prior_20260827.py", "amb")
comp = load_module(ROOT / "work" / "histrestore_first_round" / "build_semantic_compatibility_features_20260827.py", "comp")

SEEDS = [101, 202, 303, 404, 505]
HELDOUT_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001]

VARIANTS = {
    "evidence": [],
    "raw_prior": [],
    "compat_full": comp.COMPAT_NAMES,
    "compat_policy": [
        "sem_policy_match",
        "sem_policy_mismatch",
        "sem_preserve_input_match",
        "sem_deshadow_docres_match",
        "sem_binarize_binary_match",
        "sem_blend_blend_match",
    ],
    "compat_risk_strength": [
        "sem_strength_gap",
        "sem_risk_candidate_strength",
        "sem_highrisk_mild_match",
        "sem_highrisk_aggressive_conflict",
        "sem_lowrisk_aggressive_allowed",
    ],
    "compat_balanced": [
        "sem_policy_match",
        "sem_policy_mismatch",
        "sem_strength_gap",
        "sem_risk_candidate_strength",
        "sem_highrisk_mild_match",
        "sem_highrisk_aggressive_conflict",
        "sem_deshadow_docres_match",
        "sem_blend_blend_match",
    ],
}


def write_csv(path, rows):
    rows = list(rows)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def feature(row, base, cand, prior, variant):
    vals = hist.v35_feature(row, base, None, use_risk=True, use_identity=True)
    if variant != "evidence":
        qfeat = hist.qwen_feature_dict(prior)
        vals += [qfeat.get(name, 0.0) for name in hist.QWEN_FEATS]
    if variant.startswith("compat"):
        all_comp = comp.compatibility_features(cand, prior)
        vals += [all_comp[name] for name in VARIANTS[variant]]
    return vals


def utility(row, base):
    return (float(row["psnr"]) - float(base["psnr"])) + 40.0 * (float(row["ssim"]) - float(base["ssim"]))


def make_xy(grouped, order, priors, keys, variant):
    x, y, row_keys, cands = [], [], [], []
    for key in sorted(keys):
        base = grouped[key][hist.BASE_V35]
        prior = priors.get((key[0], key[1]))
        for cand in order:
            if cand not in grouped[key]:
                continue
            row = grouped[key][cand]
            x.append(feature(row, base, cand, prior, variant))
            y.append(utility(row, base))
            row_keys.append(key)
            cands.append(cand)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), row_keys, cands


def fit_models(grouped, order, priors, train_keys, variant, seeds):
    x, y, _, _ = make_xy(grouped, order, priors, train_keys, variant)
    models = []
    for seed in seeds:
        model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1)
        model.fit(x, y)
        models.append(model)
    return models


def select(grouped, order, priors, keys, variant, models, threshold=0.25):
    x, _, row_keys, cands = make_xy(grouped, order, priors, keys, variant)
    preds = np.vstack([m.predict(x) for m in models]).mean(axis=0)
    pred_map = defaultdict(list)
    for p, key, cand in zip(preds, row_keys, cands):
        pred_map[key].append((float(p), cand))
    rows = []
    for key in sorted(keys):
        ranked = sorted(pred_map[key], reverse=True)
        p, cand = ranked[0]
        if p < threshold:
            cand = hist.BASE_V35
        rows.append({**grouped[key][cand], "method": variant, "selected_candidate": cand})
    return rows


def summarize(rows, variant, protocol):
    return {
        "protocol": protocol,
        "variant": variant,
        "n": len(rows),
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "vccrp": float(np.mean([r["content_risk"] for r in rows])),
        "utility_psnr_40ssim": float(np.mean([r["psnr"] + 40.0 * r["ssim"] for r in rows])),
        "selected_counts": json.dumps(Counter([r["selected_candidate"] for r in rows]).most_common(), ensure_ascii=False),
    }


def bootstrap(vals, n_boot=20000, seed=20260827):
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot = vals[idx].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def paired_stats(a_rows, b_rows, label):
    a = {(r["dataset"], r["sample_id"]): r for r in a_rows}
    b = {(r["dataset"], r["sample_id"]): r for r in b_rows}
    keys = sorted(set(a) & set(b))
    out = []
    for metric, nice, higher in [("psnr", "PSNR", True), ("ssim", "SSIM", True), ("content_risk", "VCCRP", False)]:
        diffs = np.asarray([float(a[k][metric]) - float(b[k][metric]) for k in keys])
        lo, hi = bootstrap(diffs)
        try:
            w_p = stats.wilcoxon(diffs, zero_method="wilcox").pvalue
        except ValueError:
            w_p = np.nan
        out.append({
            "comparison": label,
            "metric": nice,
            "n": len(keys),
            "mean_delta": float(diffs.mean()),
            "ci_low": lo,
            "ci_high": hi,
            "paired_t_p": float(stats.ttest_rel([a[k][metric] for k in keys], [b[k][metric] for k in keys]).pvalue),
            "wilcoxon_p": float(w_p),
            "cohen_dz": float(diffs.mean() / diffs.std(ddof=1)) if diffs.std(ddof=1) > 0 else np.nan,
            "win_rate": float(np.mean(diffs > 0 if higher else diffs < 0)),
        })
    return out


def main():
    _, grouped, priors, order, train_keys, val_keys = amb.load_data()
    folds = {k: int(hist.hashlib.md5(f"{k[0]}:{k[1]}".encode("utf-8")).hexdigest()[:8], 16) % 5 for k in train_keys}
    cv_rows = defaultdict(list)
    for fold in range(5):
        tr = [k for k in train_keys if folds[k] != fold]
        ev = [k for k in train_keys if folds[k] == fold]
        for variant in VARIANTS:
            models = fit_models(grouped, order, priors, tr, variant, SEEDS)
            cv_rows[variant].extend(select(grouped, order, priors, ev, variant, models))
    cv_summary = [summarize(rows, variant, "train_groupcv") for variant, rows in cv_rows.items()]
    write_csv(OUT / "train_groupcv_variant_summary.csv", cv_summary)

    held = {}
    held_summary = []
    for variant in VARIANTS:
        models = fit_models(grouped, order, priors, train_keys, variant, HELDOUT_SEEDS)
        rows = select(grouped, order, priors, val_keys, variant, models)
        held[variant] = rows
        held_summary.append(summarize(rows, variant, "heldout"))
        write_jsonl(OUT / f"selected_{variant}.jsonl", rows)
    write_csv(OUT / "heldout_variant_summary.csv", held_summary)

    best_cv = max([r for r in cv_summary if r["variant"].startswith("compat") or r["variant"] == "raw_prior"], key=lambda r: (r["utility_psnr_40ssim"], r["psnr"], r["ssim"], -r["vccrp"]))
    stats_rows = []
    for variant in VARIANTS:
        if variant == "evidence":
            continue
        stats_rows += paired_stats(held[variant], held["evidence"], f"{variant} vs evidence")
        if variant != "raw_prior":
            stats_rows += paired_stats(held[variant], held["raw_prior"], f"{variant} vs raw_prior")
    write_csv(OUT / "heldout_paired_stats.csv", stats_rows)

    report = []
    report.append("# Semantic Compatibility Feature Variants")
    report.append("")
    report.append("This experiment tests smaller generic subsets of candidate-aware semantic compatibility features. The goal is to keep the compatibility prior useful without adding noisy interactions.")
    report.append("")
    report.append(f"Train-CV selected semantic variant by `PSNR + 40 SSIM`: `{best_cv['variant']}`.")
    report.append("")
    report.append("## Train Group-CV")
    report.append(pd.DataFrame(cv_summary).drop(columns=["selected_counts"]).sort_values("utility_psnr_40ssim", ascending=False).to_markdown(index=False))
    report.append("")
    report.append("## Held-Out")
    report.append(pd.DataFrame(held_summary).drop(columns=["selected_counts"]).sort_values("utility_psnr_40ssim", ascending=False).to_markdown(index=False))
    report.append("")
    report.append("## Paired Statistics")
    report.append(pd.DataFrame(stats_rows).to_markdown(index=False))
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "best_cv_variant": best_cv["variant"], "heldout": held_summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
