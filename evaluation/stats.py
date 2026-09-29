"""
Statistical comparison of paired per-question scores between the two
pipelines.

For each metric, we have N paired observations (one score per pipeline per
question, same question -> paired design). We report:

  - mean and median for each pipeline
  - mean difference (enhanced - naive) with a bootstrap 95% CI
  - Wilcoxon signed-rank test (non-parametric, appropriate for bounded
    [0,1] scores that are often non-normal) -> p-value
  - paired t-test -> p-value, as a secondary check
  - matched-pairs rank-biserial correlation as an effect size for the
    Wilcoxon test

A result is reported as a genuine "win" only if:
  - the mean difference is positive (enhanced > naive) AND
  - the Wilcoxon p-value < alpha (default 0.05)

With small N (e.g. the ~12-question demo dataset), p-values will usually NOT
reach significance -- that's expected and correct. The real benchmark needs
N ~ 100-150+ for adequate power at typical effect sizes seen in RAG
literature (Cohen's d ~ 0.3-0.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import stats


@dataclass
class PairedTestResult:
    metric: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    diff_ci_low: float
    diff_ci_high: float
    wilcoxon_p: float
    ttest_p: float
    effect_size_rank_biserial: float

    def is_significant_win_for_b(self, alpha: float = 0.05) -> bool:
        return self.mean_diff > 0 and self.wilcoxon_p < alpha

    def __str__(self) -> str:
        sig = "SIGNIFICANT" if self.wilcoxon_p < 0.05 else "not significant"
        direction = "enhanced > naive" if self.mean_diff > 0 else "enhanced < naive"
        return (
            f"{self.metric:24s} naive={self.mean_a:.3f}  enhanced={self.mean_b:.3f}  "
            f"diff={self.mean_diff:+.3f}  95% CI=[{self.diff_ci_low:+.3f}, "
            f"{self.diff_ci_high:+.3f}]  wilcoxon p={self.wilcoxon_p:.4f}  "
            f"ttest p={self.ttest_p:.4f}  effect_r={self.effect_size_rank_biserial:.3f}  "
            f"({direction}, {sig})"
        )


def _bootstrap_ci_mean_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = b - a
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = sample.mean()
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _rank_biserial_effect_size(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation, the standard effect size
    companion to the Wilcoxon signed-rank test. Ranges -1 to 1."""
    diffs = b - a
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    pos = ranks[nonzero > 0].sum()
    neg = ranks[nonzero < 0].sum()
    total = pos + neg
    if total == 0:
        return 0.0
    return float((pos - neg) / total)


def compare_metric(
    metric_name: str, scores_naive: List[float], scores_enhanced: List[float]
) -> PairedTestResult:
    a = np.asarray(scores_naive, dtype=float)
    b = np.asarray(scores_enhanced, dtype=float)
    assert len(a) == len(b) and len(a) > 0

    diff = b - a
    mean_diff = float(diff.mean())

    if np.all(diff == 0):
        wilcoxon_p = 1.0
    else:
        try:
            _, wilcoxon_p = stats.wilcoxon(a, b)
        except ValueError:
            wilcoxon_p = 1.0

    if len(a) > 1 and np.std(diff) > 0:
        _, ttest_p = stats.ttest_rel(a, b)
    else:
        ttest_p = 1.0

    ci_low, ci_high = _bootstrap_ci_mean_diff(a, b)
    effect = _rank_biserial_effect_size(a, b)

    return PairedTestResult(
        metric=metric_name,
        n=len(a),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=mean_diff,
        diff_ci_low=ci_low,
        diff_ci_high=ci_high,
        wilcoxon_p=float(wilcoxon_p),
        ttest_p=float(ttest_p),
        effect_size_rank_biserial=effect,
    )
