import numpy as np
from scipy import stats


def paired_bootstrap(diffs, n_boot=20000, seed=20260827):
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    boot = diffs[idx].mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def paired_test(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = a - b
    lo, hi = paired_bootstrap(diffs)
    try:
        wilcoxon_p = float(stats.wilcoxon(diffs, zero_method="wilcox").pvalue)
    except ValueError:
        wilcoxon_p = float("nan")
    return {
        "mean_delta": float(diffs.mean()),
        "ci95_low": lo,
        "ci95_high": hi,
        "paired_t_p": float(stats.ttest_rel(a, b).pvalue),
        "wilcoxon_p": wilcoxon_p,
        "wins": int((diffs > 1e-12).sum()),
        "losses": int((diffs < -1e-12).sum()),
        "ties": int((np.abs(diffs) <= 1e-12).sum()),
    }

