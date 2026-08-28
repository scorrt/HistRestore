from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = [
    ROOT / "results/historical537/main_results.csv",
    ROOT / "results/historical537/table_s1_paired_statistics.csv",
    ROOT / "results/mixedoc/main_results.csv",
    ROOT / "results/mixedoc/paired_bootstrap.csv",
    ROOT / "results/runtime/runtime_summary.csv",
]

def main():
    for path in TABLES:
        print(f"\n## {path.relative_to(ROOT)}")
        print(pd.read_csv(path).to_string(index=False))

if __name__ == "__main__":
    main()
