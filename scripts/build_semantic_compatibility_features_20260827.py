import csv
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "semantic_compatibility_features_20260827"
OUT.mkdir(parents=True, exist_ok=True)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hist = load_module(ROOT / "work" / "histrestore_first_round" / "build_historical537_group_split_full_audit.py", "hist")
amb = load_module(ROOT / "work" / "histrestore_first_round" / "build_ambiguity_gated_prior_20260827.py", "amb")
gate = load_module(ROOT / "work" / "histrestore_first_round" / "build_semantic_disagreement_gate_20260827.py", "gate")


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


def blend_alpha(candidate):
    m = re.search(r"_blend_(0\.\d+)", str(candidate))
    return float(m.group(1)) if m else 0.0


def candidate_strength(candidate):
    fam = gate.candidate_family(candidate)
    if fam == "input":
        return 0.0
    if fam == "classical":
        return 0.30
    if fam == "blend":
        alpha = blend_alpha(candidate)
        return 0.50 + 0.40 * max(alpha - 0.70, 0.0) / 0.25 if alpha else 0.65
    if fam in {"docres", "appearance", "binarization", "region"}:
        return 0.85
    return 0.50


def qwen_target_strength(prior):
    strength = gate.strength_value(prior)
    return {
        "none": 0.0,
        "light": 0.30,
        "medium": 0.65,
        "strong": 0.90,
    }.get(strength, 0.50)


def risk_numeric(prior):
    return {"low": 0.0, "medium": 0.5, "high": 1.0}.get(gate.risk_value(prior), 0.5)


def compatibility_features(candidate, prior):
    fam = gate.candidate_family(candidate)
    allowed = gate.prior_allowed_families(prior)
    policy = gate.policy_value(prior)
    c_strength = candidate_strength(candidate)
    q_strength = qwen_target_strength(prior)
    risk = risk_numeric(prior)
    is_mild = c_strength <= 0.35
    is_aggressive = c_strength >= 0.80
    out = {
        "sem_policy_match": float(fam in allowed),
        "sem_policy_mismatch": float(fam not in allowed),
        "sem_strength_gap": abs(c_strength - q_strength),
        "sem_strength_product": c_strength * q_strength,
        "sem_risk_candidate_strength": risk * c_strength,
        "sem_highrisk_mild_match": float(risk >= 1.0 and is_mild),
        "sem_highrisk_aggressive_conflict": float(risk >= 1.0 and is_aggressive),
        "sem_lowrisk_aggressive_allowed": float(risk <= 0.0 and is_aggressive),
        "sem_preserve_input_match": float(policy == "preserve" and fam == "input"),
        "sem_deshadow_docres_match": float(policy == "deshadow" and fam in {"docres", "blend", "region"}),
        "sem_binarize_binary_match": float(policy == "binarize" and fam == "binarization"),
        "sem_blend_blend_match": float(policy == "blend" and fam == "blend"),
    }
    return out


COMPAT_NAMES = list(compatibility_features("input", None).keys())


def feature(row, base, candidate, prior=None, mode="evidence"):
    vals = hist.v35_feature(row, base, None, use_risk=True, use_identity=True)
    if mode in {"raw_prior", "compat_prior"}:
        qfeat = hist.qwen_feature_dict(prior)
        vals += [qfeat.get(name, 0.0) for name in hist.QWEN_FEATS]
    if mode == "compat_prior":
        comp = compatibility_features(candidate, prior)
        vals += [comp[name] for name in COMPAT_NAMES]
    return vals


def utility(row, base, ssim_weight=40.0):
    return (float(row["psnr"]) - float(base["psnr"])) + ssim_weight * (float(row["ssim"]) - float(base["ssim"]))


def make_xy(grouped, order, keys, priors, mode, ssim_weight=40.0):
    x, y, row_keys, cands = [], [], [], []
    for key in sorted(keys):
        items = grouped[key]
        base = items[hist.BASE_V35]
        prior = priors.get((key[0], key[1]))
        for cand in order:
            if cand not in items:
                continue
            row = items[cand]
            x.append(feature(row, base, cand, prior, mode=mode))
            y.append(utility(row, base, ssim_weight))
            row_keys.append(key)
            cands.append(cand)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), row_keys, cands


def make_model(model_type, seed):
    if model_type == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    if model_type == "RF":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    if model_type == "GBDT":
        return GradientBoostingRegressor(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.022,
            min_samples_leaf=3,
            random_state=seed,
        )
    raise ValueError(model_type)


def train_select(grouped, order, priors, train_keys, eval_keys, mode, model_type, seed, threshold=0.25):
    x_train, y_train, _, _ = make_xy(grouped, order, train_keys, priors, mode)
    model = make_model(model_type, seed)
    model.fit(x_train, y_train)
    x_eval, _, row_keys, cands = make_xy(grouped, order, eval_keys, priors, mode)
    preds = model.predict(x_eval)
    pred_map = defaultdict(list)
    for pred, key, cand in zip(preds, row_keys, cands):
        pred_map[key].append((float(pred), cand))
    rows = []
    for key in sorted(eval_keys):
        items = grouped[key]
        ranked = sorted(pred_map[key], reverse=True)
        pred, chosen = ranked[0]
        if pred < threshold:
            chosen = hist.BASE_V35
        rows.append({**items[chosen], "method": f"{model_type}_{mode}", "selected_candidate": chosen})
    return rows


def run_train_cv(grouped, order, priors, train_keys, mode, model_type, seed, threshold=0.25):
    folds = {k: int(hist.hashlib.md5(f"{k[0]}:{k[1]}".encode("utf-8")).hexdigest()[:8], 16) % 5 for k in train_keys}
    rows = []
    for fold in range(5):
        tr = [k for k in train_keys if folds[k] != fold]
        ev = [k for k in train_keys if folds[k] == fold]
        rows.extend(train_select(grouped, order, priors, tr, ev, mode, model_type, seed + fold, threshold=threshold))
    return rows


def summarize(rows, label, seed):
    return {
        "seed": seed,
        "method": label,
        "n": len(rows),
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "vccrp": float(np.mean([r["content_risk"] for r in rows])),
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
    seeds = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001]
    model_types = ["ExtraTrees", "RF", "GBDT"]
    modes = ["evidence", "raw_prior", "compat_prior"]
    all_rows = []
    selected = {}
    for model_type in model_types:
        for seed in seeds:
            for mode in modes:
                rows = train_select(grouped, order, priors, train_keys, val_keys, mode, model_type, seed)
                key = (model_type, mode, seed)
                selected[key] = rows
                all_rows.append(summarize(rows, f"{model_type}_{mode}", seed))
                if seed == seeds[0]:
                    write_jsonl(OUT / f"selected_seed101_{model_type}_{mode}.jsonl", rows)
    write_csv(OUT / "heldout_seed_level_results.csv", all_rows)

    df = pd.DataFrame(all_rows)
    agg = []
    for method, gdf in df.groupby("method"):
        agg.append({
            "method": method,
            "seeds": int(len(gdf)),
            "psnr_mean": float(gdf["psnr"].mean()),
            "psnr_std": float(gdf["psnr"].std(ddof=1)),
            "ssim_mean": float(gdf["ssim"].mean()),
            "ssim_std": float(gdf["ssim"].std(ddof=1)),
            "vccrp_mean": float(gdf["vccrp"].mean()),
            "vccrp_std": float(gdf["vccrp"].std(ddof=1)),
        })
    write_csv(OUT / "heldout_10seed_summary.csv", agg)

    cv_rows = []
    for model_type in model_types:
        for mode in modes:
            rows = run_train_cv(grouped, order, priors, train_keys, mode, model_type, seed=101)
            cv_rows.append(summarize(rows, f"{model_type}_{mode}", 101))
    write_csv(OUT / "train_groupcv_model_mode_summary.csv", cv_rows)

    stats_rows = []
    seed0 = seeds[0]
    for model_type in model_types:
        evidence = selected[(model_type, "evidence", seed0)]
        raw = selected[(model_type, "raw_prior", seed0)]
        compat = selected[(model_type, "compat_prior", seed0)]
        stats_rows += paired_stats(raw, evidence, f"{model_type} raw_prior vs evidence")
        stats_rows += paired_stats(compat, evidence, f"{model_type} compat_prior vs evidence")
        stats_rows += paired_stats(compat, raw, f"{model_type} compat_prior vs raw_prior")
    write_csv(OUT / "paired_stats_seed101.csv", stats_rows)

    report = []
    report.append("# Semantic Compatibility Feature Optimization")
    report.append("")
    report.append("This run keeps the selector formulation unchanged but makes the semantic prior candidate-aware. Instead of appending only page-level Qwen tags to every candidate, it appends general compatibility features between the Qwen policy/strength/risk tags and each candidate family.")
    report.append("")
    report.append("Compatibility features: `" + "`, `".join(COMPAT_NAMES) + "`.")
    report.append("")
    report.append("## 10-Seed Held-Out Summary")
    report.append(pd.DataFrame(agg).sort_values("psnr_mean", ascending=False).to_markdown(index=False))
    report.append("")
    report.append("## 424-Page Train Group-CV Check")
    report.append(pd.DataFrame(cv_rows).drop(columns=["selected_counts"]).sort_values("psnr", ascending=False).to_markdown(index=False))
    report.append("")
    report.append("## Seed-101 Paired Statistics")
    report.append(pd.DataFrame(stats_rows).to_markdown(index=False))
    report.append("")
    report.append("## Interpretation")
    report.append("- `raw_prior` is the previous page-level Qwen-tag concatenation.")
    report.append("- `compat_prior` tests whether semantic information is more useful when expressed as candidate-aware evidence, matching the paper's evidence-constrained VLM story.")
    report.append("- The feature definitions are generic: they depend only on candidate family, Qwen policy, Qwen strength, and Qwen content-risk labels.")
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "summary": agg}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
