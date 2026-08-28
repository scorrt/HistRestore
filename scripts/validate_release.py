from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def close(a, b, tol=1e-3):
    return abs(float(a) - float(b)) <= tol

def main():
    hist = pd.read_csv(ROOT / "results/historical537/main_results.csv")
    vals = dict(zip(hist.method, hist.psnr))
    assert close(vals["Best fixed blend (DocRes-classical 0.85)"], 24.576)
    assert close(vals["Evidence-only HistRestore"], 24.683)
    assert close(vals["Direct structured review"], 25.006)
    assert close(vals["PSNR oracle"], 26.088)

    mix = pd.read_csv(ROOT / "results/mixedoc/main_results.csv")
    m = dict(zip(mix.method, mix.psnr))
    assert close(m["MMDIR official"], 27.520)
    assert close(m["HistRestore + MMDIR"], 27.667)

    split = pd.read_csv(ROOT / "splits/historical537_group_split_counts.csv")
    assert int(split["samples"].sum()) == 537
    assert int(split.loc[split["split"].eq("train"), "samples"].sum()) == 424
    assert int(split.loc[split["split"].eq("val"), "samples"].sum()) == 113

    ocr = json.loads((ROOT / "results/ocr/ocr_diagnostic_coverage.json").read_text())
    assert ocr["sample_count"] == 50
    assert ocr["row_count"] == 250

    runtime = pd.read_csv(ROOT / "results/runtime/runtime_summary.csv")
    assert runtime.gpu.astype(str).str.contains("A100").all()

    print("HistRestore release validation passed.")

if __name__ == "__main__":
    main()
