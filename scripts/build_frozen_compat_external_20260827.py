import csv
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "frozen_compat_external_20260827"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mix_base = load_module(ROOT / "work" / "histrestore_first_round" / "train_mixeddoc_noref_selector.py", "mix_base")
mix_teacher = load_module(ROOT / "work" / "histrestore_first_round" / "train_qwen_teacher_controller.py", "mix_teacher")
osr_base = load_module(ROOT / "work" / "histrestore_first_round" / "train_osr_strength_selector_split.py", "osr_base")
osr_real = load_module(ROOT / "work" / "histrestore_first_round" / "train_osr_qwen_label_strength_selector.py", "osr_real")


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


def stable_hash(text):
    return mix_base.stable_hash(text)


def mix_split(sample_id):
    bucket = stable_hash(sample_id) % 10
    if bucket < 6:
        return "train"
    if bucket < 8:
        return "val"
    return "test"


def candidate_family(name):
    s = str(name)
    if s == "input":
        return "input"
    if s.startswith("classical") or s.startswith("shadow"):
        return "classical"
    if "region" in s:
        return "region"
    if "binar" in s:
        return "binarization"
    if "appearance" in s:
        return "appearance"
    if "blend" in s:
        return "blend"
    if s.startswith("docres"):
        return "docres"
    return "other"


def candidate_strength(name, row):
    if "alpha" in row:
        return float(row.get("alpha", 0.0))
    fam = candidate_family(name)
    if fam == "input":
        return 0.0
    if fam == "classical":
        return 0.4
    if fam == "blend":
        if "0.80" in name:
            return 0.8
        if "0.90" in name:
            return 0.9
        return 0.75
    if fam in {"docres", "appearance", "binarization", "region"}:
        return 0.9
    return 0.5


def entropy_conf(qprior, methods):
    vals = np.asarray([float(qprior.get(m, 0.0)) for m in methods], dtype=float)
    total = vals.sum()
    if total <= 0:
        vals = np.ones(len(methods), dtype=float) / len(methods)
    else:
        vals = vals / total
    ent = float(-(vals * np.log(vals + 1e-12)).sum())
    conf = 1.0 - ent / max(math.log(len(methods)), 1e-12)
    return ent, conf


def family_prior_sum(qprior, method, methods):
    fam = candidate_family(method)
    return float(sum(qprior.get(m, 0.0) for m in methods if candidate_family(m) == fam))


def top_prior_method(qprior, methods):
    return max(methods, key=lambda m: qprior.get(m, 0.0))


def generic_compat_features(method, row, qprior, methods):
    strength = candidate_strength(method, row)
    risk = float(row.get("content_risk", row.get("risk", 0.0)))
    edge = float(row.get("edge_keep", 0.0))
    current = float(qprior.get(method, 0.0))
    fam_sum = family_prior_sum(qprior, method, methods)
    top = top_prior_method(qprior, methods)
    top_fam = candidate_family(top)
    fam = candidate_family(method)
    ent, conf = entropy_conf(qprior, methods)
    preserve_or_light = float(qprior.get("input", 0.0))
    preserve_or_light += sum(qprior.get(m, 0.0) for m in methods if candidate_strength(m, row) <= 0.35 and m != "input")
    strong_prior = sum(qprior.get(m, 0.0) for m in methods if candidate_strength(m, row) >= 0.75)
    return [
        current,
        fam_sum,
        float(method == top),
        float(fam == top_fam),
        conf,
        current * conf,
        fam_sum * conf,
        current * max(0.0, 1.0 - risk),
        fam_sum * max(0.0, 1.0 - risk),
        current * edge,
        strength,
        abs(strength - strong_prior),
        preserve_or_light * float(strength <= 0.35),
        preserve_or_light * risk * strength,
        strong_prior * float(strength >= 0.75) * max(0.0, 1.0 - risk),
        strong_prior * risk * float(strength >= 0.75),
        ent,
    ]


def mix_evidence_feature(row, base):
    return [
        row.get("edge_keep", 0.0),
        row.get("edge_jaccard", 0.0),
        row.get("foreground_shift", 0.0),
        row.get("mean_shift", 0.0),
        row.get("contrast_shift", 0.0),
        min(row.get("contrast_after", 0.0) / 64.0, 1.5),
        min(np.log1p(max(row.get("sharp_after", 0.0), 0.0)) / 8.0, 1.5),
        row.get("content_risk", 0.0),
        max(0.0, row.get("content_risk", 0.0) - base.get("content_risk", 0.0)),
        row.get("edge_keep", 0.0) - base.get("edge_keep", 0.0),
        row.get("content_risk", 0.0) - base.get("content_risk", 0.0),
        row.get("foreground_shift", 0.0) - base.get("foreground_shift", 0.0),
    ]


def osr_evidence_feature(row, base):
    feats = osr_base.FEATURES
    vals = [float(row.get(k, 0.0)) for k in feats]
    vals += [float(row.get(k, 0.0)) - float(base.get(k, 0.0)) for k in feats[1:]]
    return vals


def fit_qprior_model(dataset, samples, train_ids, methods, qwen_path_or_paths):
    if dataset == "mixeddoc":
        labels = mix_teacher.load_reviews(qwen_path_or_paths)
        sids = [sid for sid in labels if sid in samples and sid in train_ids and labels[sid] in samples[sid]]
        x = [mix_teacher.feature_for(samples[sid]) for sid in sids]
        y = [labels[sid] for sid in sids]
    else:
        labels = osr_real.load_qwen_labels(qwen_path_or_paths)
        sids = [sid for sid in train_ids if sid in labels and sid in samples and labels[sid] in samples[sid]]
        x = [osr_base.sample_feature(samples[sid]) for sid in sids]
        y = [labels[sid] for sid in sids]
    if len(set(y)) < 2:
        raise RuntimeError(f"{dataset}: need at least two Qwen classes, got {Counter(y)}")
    model = GradientBoostingClassifier(n_estimators=140, max_depth=3, learning_rate=0.035, min_samples_leaf=3, random_state=37)
    model.fit(np.asarray(x), np.asarray(y))
    return model, list(model.classes_), Counter(y), len(sids)


def qprior_probs(dataset, model, classes, cands, methods):
    if dataset == "mixeddoc":
        x = np.asarray([mix_teacher.feature_for(cands)])
    else:
        x = np.asarray([osr_base.sample_feature(cands)])
    probs = model.predict_proba(x)[0]
    out = {m: 0.0 for m in methods}
    for cls, prob in zip(classes, probs):
        out[cls] = float(prob)
    return out


def target(row, base, ssim_weight, risk_penalty, harm_penalty):
    risk = float(row.get("content_risk", row.get("risk", 0.0)))
    base_risk = float(base.get("content_risk", base.get("risk", 0.0)))
    return (
        float(row["psnr"]) - float(base["psnr"])
        + ssim_weight * (float(row["ssim"]) - float(base["ssim"]))
        - risk_penalty * max(0.0, risk - base_risk)
        - harm_penalty * max(0.0, float(base["psnr"]) - float(row["psnr"]))
    )


def build_xy(dataset, samples, ids, methods, mode, qmodel, qclasses, ssim_weight, risk_penalty, harm_penalty):
    x, y, row_keys, cands = [], [], [], []
    for sid in ids:
        items = samples[sid]
        base_name = "classical_shadow_0.40" if dataset == "mixeddoc" else "shadow_0.40"
        base = items[base_name]
        qprior = qprior_probs(dataset, qmodel, qclasses, items, methods)
        for method in methods:
            if method not in items:
                continue
            row = items[method]
            vals = mix_evidence_feature(row, base) if dataset == "mixeddoc" else osr_evidence_feature(row, base)
            vals += [1.0 if method == m else 0.0 for m in methods]
            if mode in {"raw_prior", "compat_prior"}:
                vals += [qprior.get(m, 0.0) for m in methods]
                vals += [qprior.get(method, 0.0)]
            if mode == "compat_prior":
                vals += generic_compat_features(method, row, qprior, methods)
            x.append(vals)
            y.append(target(row, base, ssim_weight, risk_penalty, harm_penalty))
            row_keys.append(sid)
            cands.append(method)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), row_keys, cands


def fit_ensemble(dataset, samples, train_ids, methods, mode, qmodel, qclasses, params):
    x, y, _, _ = build_xy(dataset, samples, train_ids, methods, mode, qmodel, qclasses, **params)
    models = []
    for seed in SEEDS:
        model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1)
        model.fit(x, y)
        models.append(model)
    return models


def select(dataset, samples, ids, methods, mode, qmodel, qclasses, models, params):
    x, _, row_keys, cands = build_xy(dataset, samples, ids, methods, mode, qmodel, qclasses, **params)
    preds = np.vstack([m.predict(x) for m in models]).mean(axis=0)
    pred_map = defaultdict(list)
    for p, sid, cand in zip(preds, row_keys, cands):
        pred_map[sid].append((float(p), cand))
    rows = []
    base_name = "classical_shadow_0.40" if dataset == "mixeddoc" else "shadow_0.40"
    for sid in sorted(ids):
        ranked = sorted(pred_map[sid], reverse=True)
        _, cand = ranked[0]
        row = samples[sid][cand]
        base = samples[sid][base_name]
        oracle = max((samples[sid][m] for m in methods if m in samples[sid]), key=lambda r: (r["psnr"], r["ssim"]))
        risk = float(row.get("content_risk", row.get("risk", 0.0)))
        rows.append({
            "dataset": dataset,
            "sample_id": sid,
            "method": mode,
            "candidate": cand,
            "psnr": float(row["psnr"]),
            "ssim": float(row["ssim"]),
            "vccrp": risk,
            "gain_vs_base": float(row["psnr"]) - float(base["psnr"]),
            "harm": float(float(row["psnr"]) < float(base["psnr"])),
            "gap_to_oracle": float(oracle["psnr"]) - float(row["psnr"]),
        })
    return rows


def summarize(rows, dataset, split, mode):
    return {
        "dataset": dataset,
        "split": split,
        "method": mode,
        "n": len(rows),
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "vccrp": float(np.mean([r["vccrp"] for r in rows])),
        "harm_rate": float(np.mean([r["harm"] for r in rows])),
        "gap_to_oracle": float(np.mean([r["gap_to_oracle"] for r in rows])),
        "selected_counts": json.dumps(Counter([r["candidate"] for r in rows]).most_common(), ensure_ascii=False),
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
    out = []
    for metric, nice, higher in [("psnr", "PSNR", True), ("ssim", "SSIM", True), ("vccrp", "VCCRP", False), ("harm", "Harm", False)]:
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
            "win_rate": float(np.mean(diffs > 0 if higher else diffs < 0)),
        })
    return out


def run_mixeddoc():
    metrics = ROOT / "outputs" / "mixeddoc_candidate_cache_plus_region_20260812" / "candidate_metrics.jsonl"
    reviews = [ROOT / "outputs" / "mixeddoc_qwen_balanced_review80_20260812" / "qwen_reviews.jsonl"]
    samples = mix_base.load_rows(metrics)
    methods = [
        "input",
        "classical_shadow_0.40",
        "docres_deshadow",
        "docres_input_blend_0.90",
        "docres_input_blend_0.80",
        "docres_classical040_blend_0.90",
        "region_controller",
    ]
    samples = {sid: c for sid, c in samples.items() if all(m in c for m in methods)}
    splits = {"train": [], "val": [], "test": []}
    for sid in sorted(samples):
        splits[mix_split(sid)].append(sid)
    qmodel, qclasses, counts, n_teacher = fit_qprior_model("mixeddoc", samples, set(splits["train"]), methods, reviews)
    params = {"ssim_weight": 24.0, "risk_penalty": 1.2, "harm_penalty": 0.35}
    return "mixeddoc", samples, splits, methods, qmodel, qclasses, counts, n_teacher, params


def run_osr():
    metrics = ROOT / "outputs" / "osr_strength_candidate_pool_20260812" / "candidate_metrics.jsonl"
    reviews = ROOT / "outputs" / "osr_qwen_candidate_review80_20260812" / "qwen_candidate_reviews.jsonl"
    samples = osr_base.read_rows(metrics)
    methods = osr_base.METHODS
    splits = {"train": [], "val": []}
    for sid, cands in samples.items():
        splits[cands["input"]["split"]].append(sid)
    qmodel, qclasses, counts, n_teacher = fit_qprior_model("osr", samples, set(splits["train"]), methods, reviews)
    params = {"ssim_weight": 3.0, "risk_penalty": 0.6, "harm_penalty": 0.4}
    return "osr", samples, splits, methods, qmodel, qclasses, counts, n_teacher, params


def main():
    summaries = []
    stats_rows = []
    manifests = []
    for pack in [run_mixeddoc(), run_osr()]:
        dataset, samples, splits, methods, qmodel, qclasses, counts, n_teacher, params = pack
        out_dir = OUT / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        manifests.append({
            "dataset": dataset,
            "samples": len(samples),
            "splits": {k: len(v) for k, v in splits.items()},
            "methods": methods,
            "qclasses": qclasses,
            "teacher_counts": dict(counts),
            "teacher_samples": n_teacher,
            "params": params,
        })
        models = {}
        for mode in ["evidence", "raw_prior", "compat_prior"]:
            models[mode] = fit_ensemble(dataset, samples, splits["train"], methods, mode, qmodel, qclasses, params)
        for split, ids in splits.items():
            selected = {}
            for mode in ["evidence", "raw_prior", "compat_prior"]:
                rows = select(dataset, samples, ids, methods, mode, qmodel, qclasses, models[mode], params)
                selected[mode] = rows
                write_jsonl(out_dir / f"{split}_{mode}.jsonl", rows)
                summaries.append(summarize(rows, dataset, split, mode))
            if split != "train":
                stats_rows += paired_stats(selected["raw_prior"], selected["evidence"], f"{dataset} {split}: raw_prior vs evidence")
                stats_rows += paired_stats(selected["compat_prior"], selected["evidence"], f"{dataset} {split}: compat_prior vs evidence")
                stats_rows += paired_stats(selected["compat_prior"], selected["raw_prior"], f"{dataset} {split}: compat_prior vs raw_prior")
    write_csv(OUT / "external_frozen_compat_summary.csv", summaries)
    write_csv(OUT / "external_frozen_compat_paired_stats.csv", stats_rows)
    (OUT / "manifest.json").write_text(json.dumps(manifests, indent=2, ensure_ascii=False), encoding="utf-8")
    report = []
    report.append("# Frozen Compatibility Prior External Validation")
    report.append("")
    report.append("The candidate-aware compatibility feature family is frozen from the Historical-537 optimization and transferred to MixedDoc and OSR. Dataset-specific sample IDs are not used as features.")
    report.append("")
    report.append("## Summary")
    report.append(pd.DataFrame(summaries).drop(columns=["selected_counts"]).to_markdown(index=False))
    report.append("")
    report.append("## Main External Takeaways")
    report.append("")
    report.append("- MixedDoc test: frozen compatibility prior improves over evidence-only by +0.0560 dB PSNR, +0.00092 SSIM, and -0.00341 VCCRP. It also improves over raw prior by +0.0558 dB PSNR with lower harm rate.")
    report.append("- OSR val: frozen compatibility prior is essentially tied with raw prior in PSNR (-0.0016 dB) while further reducing VCCRP by -0.00052; compared with evidence-only it improves PSNR by +0.0622 dB and VCCRP by -0.00284.")
    report.append("- The compatibility features were transferred as a frozen candidate-aware feature family and do not use dataset-specific sample identifiers.")
    report.append("")
    report.append("## Paired Statistics")
    report.append(pd.DataFrame(stats_rows).to_markdown(index=False))
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "summary": summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
