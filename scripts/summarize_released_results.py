"""Print a compact summary of released CSV result tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


TABLES = [
    ROOT / "results" / "historical537" / "historical537_group_main_results.csv",
    ROOT / "results" / "mixedoc" / "main_sota_table_mixeddoc.csv",
    ROOT / "results" / "mixedoc" / "mmdir_augmented_bootstrap.csv",
    ROOT / "results" / "runtime_summary.csv",
]


def main() -> None:
    for path in TABLES:
        print(f"\n## {path.relative_to(ROOT)}")
        if not path.exists():
            print("missing")
            continue
        df = pd.read_csv(path)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
