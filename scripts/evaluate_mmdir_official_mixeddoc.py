import argparse
import csv
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from evaluate_mixeddoc_fast_docres_fusion import evidence, limit_long_side, psnr, resize_to, ssim_gray  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def index_zip(zf):
    out = {}
    for name in zf.namelist():
        suffix = Path(name).suffix.lower()
        if suffix in IMAGE_EXTS:
            out[Path(name).stem] = name
    return out


def imread_zip(zf, name):
    data = np.frombuffer(zf.read(name), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def summarize_metric_rows(rows, method_name=None, classical_by_id=None):
    vals = [r for r in rows if method_name is None or r.get("method") == method_name]
    if not vals:
        return None
    risks = [r.get("content_risk", r.get("risk", r.get("mean_content_risk"))) for r in vals]
    risks = [v for v in risks if v is not None]
    harm = []
    if classical_by_id:
        for r in vals:
            sid = r["sample_id"]
            if sid in classical_by_id:
                harm.append(float(r["psnr"] < classical_by_id[sid]))
    elif vals and "harm" in vals[0]:
        harm = [float(r.get("harm", 0.0)) for r in vals]
    return {
        "method": method_name or vals[0].get("method", "unknown"),
        "n": len(vals),
        "psnr": float(np.mean([r["psnr"] for r in vals])),
        "ssim": float(np.mean([r["ssim"] for r in vals])),
        "content_risk": float(np.mean(risks)) if risks else float("nan"),
        "harm_rate_vs_classical040": float(np.mean(harm)) if harm else float("nan"),
    }


def load_candidate_metrics(path, sample_ids):
    grouped = defaultdict(dict)
    for r in read_jsonl(path):
        sid = r["sample_id"]
        if sid in sample_ids:
            grouped[sid][r["method"]] = r
    return grouped


def summarize_candidate_method(grouped, method):
    rows = []
    classical = {}
    for sid, methods in grouped.items():
        if "classical_shadow_0.40" in methods:
            classical[sid] = methods["classical_shadow_0.40"]["psnr"]
        if method in methods:
            rows.append(methods[method])
    return summarize_metric_rows(rows, method, classical)


def summarize_selected(path, label, sample_ids, classical_by_id):
    rows = []
    for r in read_jsonl(path):
        if r["sample_id"] in sample_ids:
            rows.append(
                {
                    "sample_id": r["sample_id"],
                    "method": label,
                    "psnr": r["psnr"],
                    "ssim": r["ssim"],
                    "content_risk": r.get("risk", r.get("content_risk", float("nan"))),
                    "harm": r.get("harm"),
                }
            )
    return summarize_metric_rows(rows, label, classical_by_id)


def eval_official(manifest, clean_zip, degraded_zip, official_zip, include_ids, max_side):
    rows = []
    skipped = []
    with zipfile.ZipFile(clean_zip) as zclean, zipfile.ZipFile(degraded_zip) as zdegraded, zipfile.ZipFile(official_zip) as zofficial:
        clean_idx = index_zip(zclean)
        degraded_idx = index_zip(zdegraded)
        official_idx = index_zip(zofficial)
        for rec in manifest:
            sid = rec["sample_id"]
            if include_ids and sid not in include_ids:
                continue
            original = rec["original_sample_id"]
            if original not in clean_idx or original not in degraded_idx or original not in official_idx:
                skipped.append({"sample_id": sid, "original_sample_id": original})
                continue
            clean = imread_zip(zclean, clean_idx[original])
            before = imread_zip(zdegraded, degraded_idx[original])
            official = imread_zip(zofficial, official_idx[original])
            if clean is None or before is None or official is None:
                skipped.append({"sample_id": sid, "original_sample_id": original, "reason": "decode_failed"})
                continue
            before = resize_to(before, clean)
            official = resize_to(official, clean)
            clean = limit_long_side(clean, max_side)
            before = limit_long_side(before, max_side)
            official = limit_long_side(official, max_side)
            for method, image in [("input_degraded", before), ("mmdir_official", official)]:
                ev = evidence(before, image)
                rows.append(
                    {
                        "sample_id": sid,
                        "original_sample_id": original,
                        "method": method,
                        "psnr": psnr(clean, image),
                        "ssim": ssim_gray(clean, image),
                        **ev,
                    }
                )
    return rows, skipped


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--clean-zip", required=True)
    parser.add_argument("--degraded-zip", required=True)
    parser.add_argument("--official-zip", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--no-qwen", required=True)
    parser.add_argument("--qwen", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split-ids", default="")
    parser.add_argument("--max-side", type=int, default=960)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    include_ids = set()
    if args.split_ids:
        include_ids = {r["sample_id"] for r in read_jsonl(args.split_ids)}
    else:
        include_ids = {r["sample_id"] for r in manifest}

    official_rows, skipped = eval_official(
        manifest,
        args.clean_zip,
        args.degraded_zip,
        args.official_zip,
        include_ids,
        args.max_side,
    )
    with (out / "mmdir_official_per_sample.jsonl").open("w", encoding="utf-8") as f:
        for r in official_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if official_rows:
        write_csv(out / "mmdir_official_per_sample.csv", official_rows)

    sample_ids = {r["sample_id"] for r in official_rows}
    grouped_candidates = load_candidate_metrics(args.candidate_metrics, sample_ids)
    classical_by_id = {
        sid: methods["classical_shadow_0.40"]["psnr"]
        for sid, methods in grouped_candidates.items()
        if "classical_shadow_0.40" in methods
    }
    summary_rows = []
    for method in ["input_degraded", "mmdir_official"]:
        summary_rows.append(summarize_metric_rows(official_rows, method, classical_by_id))
    for method in [
        "classical_shadow_0.40",
        "docres_deshadow",
        "docres_input_blend_0.80",
        "docres_input_blend_0.90",
        "docres_classical040_blend_0.90",
        "region_controller",
    ]:
        row = summarize_candidate_method(grouped_candidates, method)
        if row:
            summary_rows.append(row)
    summary_rows.append(summarize_selected(args.no_qwen, "histrestore_no_qwen", sample_ids, classical_by_id))
    summary_rows.append(summarize_selected(args.qwen, "histrestore_qwen_prior", sample_ids, classical_by_id))
    summary_rows = [r for r in summary_rows if r is not None]

    write_csv(out / "same_protocol_sota_table.csv", summary_rows)
    summary = {
        "protocol": "MixedDoc official MMDIR predictions evaluated with the same 960px PSNR/SSIM/content-risk protocol and the HistRestore held-out test IDs.",
        "n_samples": len(sample_ids),
        "skipped": skipped,
        "summary": summary_rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    display = [
        "input_degraded",
        "mmdir_official",
        "classical_shadow_0.40",
        "docres_deshadow",
        "region_controller",
        "histrestore_no_qwen",
        "histrestore_qwen_prior",
    ]
    rows_by_method = {r["method"]: r for r in summary_rows}
    plot_rows = [rows_by_method[m] for m in display if m in rows_by_method]
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    x = np.arange(len(plot_rows))
    bars = ax.bar(x, [r["psnr"] for r in plot_rows], color=["#6b7280", "#dc2626", "#64748b", "#2563eb", "#0d9488", "#f97316", "#16a34a"][: len(plot_rows)])
    ax.set_xticks(x)
    ax.set_xticklabels([r["method"] for r in plot_rows], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("PSNR / dB")
    ax.set_title("MixedDoc Same-Protocol SOTA Comparison")
    ax.grid(axis="y", alpha=0.22)
    for bar, row in zip(bars, plot_rows):
        ax.text(bar.get_x() + bar.get_width() / 2, row["psnr"] + 0.04, f"{row['psnr']:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "same_protocol_sota_table.png", dpi=180)
    print(json.dumps({"out": str(out), "n_samples": len(sample_ids), "summary": summary_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
