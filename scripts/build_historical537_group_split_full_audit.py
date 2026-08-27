import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


ROOT = Path(".")
OUT = ROOT / "outputs" / "historical537_group_split_full_20260817"
OUT.mkdir(parents=True, exist_ok=True)

V34_METRICS = ROOT / "outputs" / "histrestore_v34_ranked_utility_20260810" / "metrics.jsonl"
V35_CANDIDATES = ROOT / "outputs" / "histrestore_v35_lite_aligned_pool_20260810" / "candidate_metrics.jsonl"
QWEN_PRIOR = ROOT / "outputs" / "qwen_degradation_prior_full_20260810.jsonl"

BASE_V34 = "shadow_0.40"
BASE_V35 = "classical_shadow_0.40"
V34_METHODS = [
    "shadow_0.40",
    "docres_deshadow",
    "docres_blend_alpha_0.85",
    "docres_appearance",
    "docres_binarization",
]
FIXED_MAIN = [
    ("input", "Input"),
    (BASE_V35, "Best Classical / shadow-0.40"),
    ("docres_deshadow", "DocRes deshadow"),
    ("docres_classical040_blend_0.85", "Best fixed blend / DocRes-classical 0.85"),
    ("docres_appearance", "DocRes appearance"),
    ("docres_binarization", "DocRes binarization"),
]

NUM_FEATS = [
    "fg_ratio",
    "near_binary_ratio",
    "midtone_ratio",
    "gray_entropy",
    "illum_before",
    "contrast_before",
    "sharp_before",
    "illum_reduction",
    "contrast_after",
    "sharp_after",
    "edge_jaccard",
    "edge_keep",
    "foreground_shift",
    "mean_shift",
    "contrast_shift",
    "content_risk",
]
V35_NUM_FEATS = NUM_FEATS + [
    "op_classical_strength",
    "op_docres_alpha",
    "op_bin_alpha",
    "op_appearance_alpha",
    "op_contrast_strength",
    "op_sharp_strength",
    "op_count",
    "family_identity",
    "family_classical",
    "family_docres",
    "family_blend",
    "family_binary",
    "family_appearance",
    "family_refine",
]
DELTA_FEATS = [
    "illum_reduction",
    "contrast_after",
    "sharp_after",
    "edge_jaccard",
    "edge_keep",
    "foreground_shift",
    "mean_shift",
    "contrast_shift",
    "content_risk",
]
QWEN_FEATS = [
    "qwen_deg_already_clean",
    "qwen_deg_shadow",
    "qwen_deg_binarization_needed",
    "qwen_deg_appearance_noise",
    "qwen_deg_complex",
    "qwen_policy_preserve",
    "qwen_policy_deshadow",
    "qwen_policy_binarize",
    "qwen_policy_blend",
    "qwen_strength_none",
    "qwen_strength_light",
    "qwen_strength_medium",
    "qwen_strength_strong",
    "qwen_risk_low",
    "qwen_risk_medium",
    "qwen_risk_high",
]


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_page_id(sample_id):
    value = re.sub(r"_v\d+$", "", str(sample_id))
    value = re.sub(r"^\d{6}_", "", value)
    return value


def group_split(dataset, sample_id):
    key = f"{dataset}:{source_page_id(sample_id)}"
    return "val" if int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % 5 == 0 else "train"


def attach_group_split(row):
    out = dict(row)
    out["old_split"] = out.get("split", "")
    out["source_page_id"] = source_page_id(out["sample_id"])
    out["split"] = group_split(out["dataset"], out["sample_id"])
    return out


def qwen_feature_dict(prior):
    out = {name: 0.0 for name in QWEN_FEATS}
    if prior:
        out[f"qwen_deg_{prior.get('degradation', 'complex')}"] = 1.0
        out[f"qwen_policy_{prior.get('policy', 'blend')}"] = 1.0
        out[f"qwen_strength_{prior.get('strength', 'medium')}"] = 1.0
        out[f"qwen_risk_{prior.get('content_risk', 'medium')}"] = 1.0
    return {name: float(out.get(name, 0.0)) for name in QWEN_FEATS}


def load_qwen_priors(path):
    return {(row["dataset"], row["sample_id"]): row for row in read_jsonl(path)}


def group_rows(rows, base_candidate):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["sample_id"], row["split"]), {})[row["candidate"]] = row
    return {key: val for key, val in grouped.items() if base_candidate in val}


def summarize(rows, label=None):
    return {
        "method": label or (rows[0]["method"] if rows else ""),
        "n": len(rows),
        "psnr": float(np.mean([r["psnr"] for r in rows])) if rows else np.nan,
        "ssim": float(np.mean([r["ssim"] for r in rows])) if rows else np.nan,
        "vccrp": float(np.mean([r["content_risk"] for r in rows])) if rows else np.nan,
    }


def bootstrap_diff(a_rows, b_rows, metric, n_boot=10000, seed=20260817):
    a = {(r["dataset"], r["sample_id"]): float(r[metric]) for r in a_rows}
    b = {(r["dataset"], r["sample_id"]): float(r[metric]) for r in b_rows}
    keys = sorted(set(a) & set(b))
    vals = np.asarray([a[k] - b[k] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot = vals[idx].mean(axis=1)
    return {
        "metric": "VCCRP" if metric == "content_risk" else metric.upper(),
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "win_rate": float(np.mean(vals < 0.0 if metric == "content_risk" else vals > 0.0)),
    }


def v34_feature(row, base, candidate, use_risk=True, use_identity=True):
    nums = NUM_FEATS if use_risk else [x for x in NUM_FEATS if x != "content_risk"]
    deltas = DELTA_FEATS if use_risk else [x for x in DELTA_FEATS if x != "content_risk"]
    vals = [float(row.get(name, 0.0)) for name in nums]
    vals += [float(row.get(name, 0.0) - base.get(name, 0.0)) for name in deltas]
    if use_identity:
        vals += [1.0 if candidate == item else 0.0 for item in V34_METHODS]
    return vals


def train_select_v34(grouped, use_risk=True, use_identity=True, threshold=0.50, ssim_weight=40.0):
    x_rows, y_rows, splits, keys, cands = [], [], [], [], []
    for key, items in sorted(grouped.items()):
        if not all(m in items for m in V34_METHODS):
            continue
        base = items[BASE_V34]
        for cand in V34_METHODS:
            row = items[cand]
            x_rows.append(v34_feature(row, base, cand, use_risk, use_identity))
            y_rows.append((row["psnr"] - base["psnr"]) + ssim_weight * (row["ssim"] - base["ssim"]))
            splits.append(key[2])
            keys.append(key)
            cands.append(cand)
    x = np.asarray(x_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)
    splits = np.asarray(splits)
    model = GradientBoostingRegressor(n_estimators=180, max_depth=4, learning_rate=0.025, random_state=11)
    model.fit(x[splits == "train"], y[splits == "train"])
    pred_map = defaultdict(list)
    for pred, key, cand in zip(model.predict(x), keys, cands):
        pred_map[key].append((float(pred), cand))
    selected = []
    per_sample = []
    for key, items in sorted(grouped.items()):
        ranked = sorted(pred_map[key], reverse=True)
        pred_gain, chosen = ranked[0]
        if pred_gain < threshold:
            chosen = BASE_V34
        oracle = max([items[c] for c in V34_METHODS], key=lambda r: (r["psnr"], r["ssim"]))
        selected.append({**items[chosen], "method": "histrestore_v34_group_noqwen", "selected_candidate": chosen})
        per_sample.append({
            "dataset": key[0],
            "sample_id": key[1],
            "split": key[2],
            "selected_candidate": chosen,
            "oracle_candidate": oracle["candidate"],
            "predicted_gain": pred_gain,
            "actual_delta_psnr": items[chosen]["psnr"] - items[BASE_V34]["psnr"],
            "oracle_gap_psnr": oracle["psnr"] - items[chosen]["psnr"],
        })
    return selected, per_sample, model


def v35_feature(row, base, qfeat=None, use_risk=True, use_identity=True):
    nums = V35_NUM_FEATS if use_risk else [x for x in V35_NUM_FEATS if x != "content_risk"]
    deltas = DELTA_FEATS if use_risk else [x for x in DELTA_FEATS if x != "content_risk"]
    vals = [float(row.get(name, 0.0)) for name in nums]
    vals += [float(row.get(name, 0.0) - base.get(name, 0.0)) for name in deltas]
    if not use_identity:
        # Remove operation-family identity from the feature vector by zeroing it,
        # while preserving operation magnitudes for a fair observable-evidence check.
        offset = len(NUM_FEATS if use_risk else [x for x in NUM_FEATS if x != "content_risk"])
        for idx, name in enumerate(nums):
            if name.startswith("family_"):
                vals[idx] = 0.0
    if qfeat is not None:
        vals += [qfeat.get(name, 0.0) for name in QWEN_FEATS]
    return vals


def train_select_v35(grouped, priors=None, use_risk=True, use_identity=True, threshold=0.25, ssim_weight=40.0):
    x_rows, y_rows, splits, keys, cands = [], [], [], [], []
    for key, items in sorted(grouped.items()):
        base = items[BASE_V35]
        qfeat = qwen_feature_dict(priors.get((key[0], key[1]))) if priors is not None else None
        for cand, row in sorted(items.items()):
            x_rows.append(v35_feature(row, base, qfeat, use_risk, use_identity))
            y_rows.append((row["psnr"] - base["psnr"]) + ssim_weight * (row["ssim"] - base["ssim"]))
            splits.append(key[2])
            keys.append(key)
            cands.append(cand)
    x = np.asarray(x_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)
    splits = np.asarray(splits)
    model = GradientBoostingRegressor(
        n_estimators=220,
        max_depth=4,
        learning_rate=0.022,
        min_samples_leaf=3,
        random_state=17 if priors is None else 19,
    )
    model.fit(x[splits == "train"], y[splits == "train"])
    pred_map = defaultdict(list)
    for pred, key, cand in zip(model.predict(x), keys, cands):
        pred_map[key].append((float(pred), cand))
    selected = []
    per_sample = []
    method = "histrestore_v35_group_qwen_prior" if priors is not None else "histrestore_v35_group_noqwen"
    for key, items in sorted(grouped.items()):
        ranked = sorted(pred_map[key], reverse=True)
        pred_gain, chosen = ranked[0]
        if pred_gain < threshold:
            chosen = BASE_V35
        oracle = max(items.values(), key=lambda r: (r["psnr"], r["ssim"]))
        selected.append({**items[chosen], "method": method, "selected_candidate": chosen})
        per_sample.append({
            "dataset": key[0],
            "sample_id": key[1],
            "split": key[2],
            "selected_candidate": chosen,
            "oracle_candidate": oracle["candidate"],
            "predicted_gain": pred_gain,
            "pred_margin": ranked[0][0] - ranked[1][0] if len(ranked) > 1 else pred_gain,
            "actual_delta_psnr": items[chosen]["psnr"] - items[BASE_V35]["psnr"],
            "oracle_gap_psnr": oracle["psnr"] - items[chosen]["psnr"],
            "selected_content_risk": items[chosen]["content_risk"],
            "oracle_content_risk": oracle["content_risk"],
        })
    return selected, per_sample, model


def system_feature(row, base, peer, system_name, qfeat):
    vals = [float(row.get(name, 0.0)) for name in NUM_FEATS]
    vals += [float(row.get(name, 0.0) - base.get(name, 0.0)) for name in DELTA_FEATS]
    vals += [float(row.get(name, 0.0) - peer.get(name, 0.0)) for name in DELTA_FEATS]
    vals += [1.0 if system_name == "v34_noqwen" else 0.0, 1.0 if system_name == "v35_qwen_prior" else 0.0]
    vals += [qfeat.get(name, 0.0) for name in QWEN_FEATS]
    return vals


def train_select_full(v34_selected, qwen_selected, v35_grouped, priors, threshold=0.25, ssim_weight=40.0):
    v34 = {(r["dataset"], r["sample_id"], r["split"]): r for r in v34_selected}
    qwen = {(r["dataset"], r["sample_id"], r["split"]): r for r in qwen_selected}
    base = {key: items[BASE_V35] for key, items in v35_grouped.items()}
    x_rows, y_rows, splits, keys, systems = [], [], [], [], []
    for key in sorted(set(v34) & set(qwen) & set(base)):
        qfeat = qwen_feature_dict(priors.get((key[0], key[1])))
        for name, row, peer in [("v34_noqwen", v34[key], qwen[key]), ("v35_qwen_prior", qwen[key], v34[key])]:
            x_rows.append(system_feature(row, base[key], peer, name, qfeat))
            y_rows.append((row["psnr"] - base[key]["psnr"]) + ssim_weight * (row["ssim"] - base[key]["ssim"]))
            splits.append(key[2])
            keys.append(key)
            systems.append(name)
    x = np.asarray(x_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)
    splits = np.asarray(splits)
    model = GradientBoostingRegressor(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.035,
        min_samples_leaf=5,
        random_state=31,
    )
    model.fit(x[splits == "train"], y[splits == "train"])
    pred_map = defaultdict(list)
    for pred, key, system in zip(model.predict(x), keys, systems):
        pred_map[key].append((float(pred), system))
    selected = []
    per_sample = []
    for key in sorted(pred_map):
        ranked = sorted(pred_map[key], reverse=True)
        system = ranked[0][1]
        if ranked[0][0] < threshold:
            system = "v34_noqwen"
        chosen = v34[key] if system == "v34_noqwen" else qwen[key]
        oracle = max([v34[key], qwen[key]], key=lambda r: (r["psnr"], r["ssim"]))
        selected.append({**chosen, "method": "histrestore_group_full", "selected_system": system})
        per_sample.append({
            "dataset": key[0],
            "sample_id": key[1],
            "split": key[2],
            "selected_system": system,
            "oracle_system": "v34_noqwen" if oracle is v34[key] else "v35_qwen_prior",
            "v34_psnr": v34[key]["psnr"],
            "qwen_prior_psnr": qwen[key]["psnr"],
            "selected_psnr": chosen["psnr"],
            "oracle_psnr": oracle["psnr"],
            "selected_minus_v34_psnr": chosen["psnr"] - v34[key]["psnr"],
        })
    pair_oracle = []
    for key in sorted(set(v34) & set(qwen)):
        pair_oracle.append({**max([v34[key], qwen[key]], key=lambda r: (r["psnr"], r["ssim"])), "method": "oracle_noqwen_qwen_pair"})
    return selected, per_sample, pair_oracle, model


def val_only(rows):
    return [r for r in rows if r["split"] == "val"]


def by_candidate(grouped, candidate):
    return [{**items[candidate], "method": candidate} for items in grouped.values() if candidate in items]


def candidate_family(name):
    name = str(name)
    if name == "input":
        return "input"
    if name == "docres_deshadow":
        return "docres_direct"
    if "shadow" in name:
        return "classical"
    if "classical040_blend" in name or name == "docres_blend_alpha_0.85":
        return "classical_guided_blend"
    if "input_blend" in name:
        return "input_blend"
    if "binary" in name or "binarization" in name:
        return "binary"
    if "appearance" in name:
        return "appearance"
    return "other"


def write_selection_profile(named_rows):
    rows = []
    for label, selected in named_rows.items():
        cur = val_only(selected)
        families = Counter(candidate_family(r.get("selected_candidate", r["candidate"])) for r in cur)
        cands = Counter(r.get("selected_candidate", r["candidate"]) for r in cur)
        n = len(cur)
        row = {"setting": label, "n": n}
        for fam in ["input", "classical", "docres_direct", "input_blend", "classical_guided_blend", "binary", "appearance", "other"]:
            row[f"{fam}_rate"] = float(families[fam] / n) if n else np.nan
        row["top_candidates"] = json.dumps(dict(cands.most_common(8)), ensure_ascii=False)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "historical537_group_selection_profile.csv", index=False)


def write_split_counts(rows):
    seen = {}
    for r in rows:
        key = (r["dataset"], r["sample_id"])
        seen[key] = (r["dataset"], r["source_page_id"], r["split"])
    agg = Counter((dataset, split) for dataset, _, split in seen.values())
    source_seen = {}
    for dataset, source, split in seen.values():
        source_seen[(dataset, source)] = split
    source_agg = Counter((dataset, split) for (dataset, _), split in source_seen.items())
    out_rows = []
    for dataset in sorted({x[0] for x in agg}):
        for split in ["train", "val"]:
            out_rows.append({
                "dataset": dataset,
                "split": split,
                "samples": agg[(dataset, split)],
                "source_pages": source_agg[(dataset, split)],
            })
    pd.DataFrame(out_rows).to_csv(OUT / "historical537_group_split_counts.csv", index=False)
    leakage_rows = []
    by_source = defaultdict(list)
    for r in rows:
        by_source[(r["dataset"], r["source_page_id"])].append((r["sample_id"], r["split"]))
    for (dataset, source), vals in sorted(by_source.items()):
        splits = sorted({split for _, split in vals})
        leakage_rows.append({
            "dataset": dataset,
            "source_page_id": source,
            "n_samples": len({sid for sid, _ in vals}),
            "splits": ",".join(splits),
            "cross_split": len(splits) > 1,
        })
    pd.DataFrame(leakage_rows).to_csv(OUT / "historical537_source_page_leakage_check.csv", index=False)


def write_report(main, ablation, boot):
    report = [
        "# Historical-537 Source-Page Group Split Audit",
        "",
        "Protocol: the split key is `dataset:source_page_id`, where `_vXX` synthetic variants and leading numeric prefixes are removed before hashing. This prevents variants from the same source page from appearing in both train and held-out validation.",
        "",
        "## Main Results on Group-Val",
        main.to_markdown(index=False),
        "",
        "## Core Ablation on Group-Val",
        ablation.to_markdown(index=False),
        "",
        "## Paired Bootstrap",
        boot.to_markdown(index=False),
        "",
        "## Audit Decision",
        "- Use this 113-sample source-page group-val table as the Historical-537 main result.",
        "- For the paper, define Full HistRestore as the expanded candidate selector with structured evidence and Qwen prior. The later post-hoc system-fusion experiment is weaker than Qwen-prior on this split and should be retired to a diagnostic note.",
        "- Retire old Historical-537 val105 tables from the main paper because their sample-level split may mix source-page variants across train and validation.",
        "- Do not claim a significant Historical-537 VCCRP reduction from the main table. On this split, the reliable claim is PSNR/SSIM improvement; content preservation should be supported by OCR, glyph, layout, and Qwen-budget evidence.",
        "- Keep MixedDoc full test, OSR val46, OCR/glyph/layout consistency, Qwen budget, and runtime as supporting or diagnostic experiments unless they use full-dataset protocols.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main():
    v34_rows = [attach_group_split(r) for r in read_jsonl(V34_METRICS)]
    v35_rows = [attach_group_split(r) for r in read_jsonl(V35_CANDIDATES)]
    priors = load_qwen_priors(QWEN_PRIOR)

    write_split_counts(v35_rows)
    v34_grouped = group_rows(v34_rows, BASE_V34)
    v35_grouped = group_rows(v35_rows, BASE_V35)

    v34_noqwen, v34_per, _ = train_select_v34(v34_grouped, use_risk=True, use_identity=True)
    v34_no_risk, _, _ = train_select_v34(v34_grouped, use_risk=False, use_identity=True)
    v34_no_identity, _, _ = train_select_v34(v34_grouped, use_risk=True, use_identity=False)
    v35_noqwen, v35_noqwen_per, _ = train_select_v35(v35_grouped, priors=None)
    v35_qwen, v35_qwen_per, _ = train_select_v35(v35_grouped, priors=priors)
    full, full_per, pair_oracle, _ = train_select_full(v34_noqwen, v35_qwen, v35_grouped, priors)
    expanded_oracle = [{**max(items.values(), key=lambda r: (r["psnr"], r["ssim"])), "method": "oracle_expanded_pool"} for items in v35_grouped.values()]

    write_jsonl(OUT / "historical537_group_v34_noqwen_selected.jsonl", v34_noqwen)
    write_jsonl(OUT / "historical537_group_v35_noqwen_selected.jsonl", v35_noqwen)
    write_jsonl(OUT / "historical537_group_v35_qwen_prior_selected.jsonl", v35_qwen)
    write_jsonl(OUT / "historical537_group_full_selected.jsonl", full)
    write_jsonl(OUT / "historical537_group_expanded_oracle.jsonl", expanded_oracle)
    write_jsonl(OUT / "historical537_group_pair_oracle.jsonl", pair_oracle)
    write_jsonl(OUT / "historical537_group_v34_noqwen_per_sample.jsonl", v34_per)
    write_jsonl(OUT / "historical537_group_v35_qwen_prior_per_sample.jsonl", v35_qwen_per)
    write_jsonl(OUT / "historical537_group_full_per_sample.jsonl", full_per)

    fixed_rows = {label: by_candidate(v35_grouped, cand) for cand, label in FIXED_MAIN}
    main_rows = [summarize(val_only(rows), label) for label, rows in fixed_rows.items()]
    main_rows += [
        summarize(val_only(v34_noqwen), "No-Qwen HistRestore / v34 fixed pool"),
        summarize(val_only(v35_noqwen), "No-Qwen HistRestore / v35 expanded pool"),
        summarize(val_only(v35_qwen), "Qwen-prior HistRestore / v35 expanded pool"),
        summarize(val_only(full), "Full HistRestore"),
        summarize(val_only(pair_oracle), "Oracle / No-Qwen-Qwen system pair"),
        summarize(val_only(expanded_oracle), "Oracle / expanded candidate pool"),
    ]
    main_table = pd.DataFrame(main_rows)
    main_table.to_csv(OUT / "historical537_group_main_results.csv", index=False)

    ablation_rows = [
        summarize(val_only(fixed_rows["DocRes deshadow"]), "Fixed DocRes"),
        summarize(val_only(v34_no_risk), "Selector without risk evidence"),
        summarize(val_only(v34_no_identity), "Selector without candidate identity"),
        summarize(val_only(v34_noqwen), "No-Qwen HistRestore / fixed candidate bank"),
        summarize(val_only(v35_noqwen), "No-Qwen HistRestore / expanded candidate bank"),
        summarize(val_only(v35_qwen), "Full HistRestore / + Qwen prior"),
        summarize(val_only(full), "Retired post-hoc system fusion"),
    ]
    ablation_table = pd.DataFrame(ablation_rows)
    ablation_table.to_csv(OUT / "historical537_group_core_ablation.csv", index=False)

    named = {
        "DocRes deshadow": fixed_rows["DocRes deshadow"],
        "Best fixed blend": fixed_rows["Best fixed blend / DocRes-classical 0.85"],
        "No-Qwen HistRestore": v34_noqwen,
        "No-Qwen v35 expanded": v35_noqwen,
        "Full HistRestore": v35_qwen,
        "Retired post-hoc system fusion": full,
        "Expanded oracle": expanded_oracle,
    }
    boot_rows = []
    comparisons = [
        ("No-Qwen vs DocRes", "No-Qwen HistRestore", "DocRes deshadow"),
        ("Full HistRestore vs No-Qwen fixed pool", "Full HistRestore", "No-Qwen HistRestore"),
        ("Full HistRestore vs No-Qwen expanded pool", "Full HistRestore", "No-Qwen v35 expanded"),
        ("Full HistRestore vs best fixed blend", "Full HistRestore", "Best fixed blend"),
        ("Full vs No-Qwen", "Full HistRestore", "No-Qwen HistRestore"),
        ("Full vs DocRes", "Full HistRestore", "DocRes deshadow"),
        ("Retired fusion vs Full HistRestore", "Retired post-hoc system fusion", "Full HistRestore"),
        ("No-Qwen v35 expanded vs No-Qwen v34 fixed", "No-Qwen v35 expanded", "No-Qwen HistRestore"),
        ("Full vs expanded oracle", "Full HistRestore", "Expanded oracle"),
    ]
    for label, a, b in comparisons:
        for metric in ["psnr", "ssim", "content_risk"]:
            boot_rows.append({"comparison": label, **bootstrap_diff(val_only(named[a]), val_only(named[b]), metric)})
    boot_table = pd.DataFrame(boot_rows)
    boot_table.to_csv(OUT / "historical537_group_paired_bootstrap.csv", index=False)

    write_selection_profile(named)
    paper_main = pd.DataFrame([
        summarize(val_only(fixed_rows["Input"]), "Input"),
        summarize(val_only(fixed_rows["Best Classical / shadow-0.40"]), "Classical shadow-0.40"),
        summarize(val_only(fixed_rows["DocRes deshadow"]), "DocRes deshadow"),
        summarize(val_only(fixed_rows["Best fixed blend / DocRes-classical 0.85"]), "Best fixed blend"),
        summarize(val_only(v35_noqwen), "HistRestore without Qwen"),
        summarize(val_only(v35_qwen), "Full HistRestore"),
        summarize(val_only(expanded_oracle), "Oracle expanded pool"),
    ])
    paper_main.to_csv(OUT / "paper_recommended_historical537_main.csv", index=False)
    paper_boot = boot_table[boot_table["comparison"].isin([
        "Full HistRestore vs No-Qwen expanded pool",
        "Full HistRestore vs best fixed blend",
        "Full vs DocRes",
        "Full vs expanded oracle",
    ])].copy()
    paper_boot.to_csv(OUT / "paper_recommended_historical537_bootstrap.csv", index=False)
    all_main_rows = []
    for label, rows in named.items():
        for row in val_only(rows):
            all_main_rows.append({**row, "audit_method": label})
    write_jsonl(OUT / "historical537_group_main_per_sample_long.jsonl", all_main_rows)

    manifest = {
        "protocol": "source-page group split",
        "split_key": "md5(dataset + ':' + source_page_id) % 5 == 0",
        "source_page_id_rule": "remove leading six-digit index and trailing _vXX variant suffix",
        "candidate_pool_rows": len(v35_rows),
        "samples": len(v35_grouped),
        "candidate_count_per_sample": Counter(len(v) for v in v35_grouped.values()),
        "qwen_prior_file": str(QWEN_PRIOR),
        "main_result_csv": str(OUT / "historical537_group_main_results.csv"),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(main_table, ablation_table, boot_table)
    print(json.dumps({"out": str(OUT), "main": main_rows, "ablation": ablation_rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
