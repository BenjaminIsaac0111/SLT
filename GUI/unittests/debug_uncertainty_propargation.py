from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from GUI.models.Annotation import Annotation


# -----------------------------------------------------------------------------
# Data container
# -----------------------------------------------------------------------------

@dataclass
class UncertaintyDebugStats:
    n: int
    mean_orig: float
    std_orig: float
    mean_adj: float
    std_adj: float
    pearson_r: float
    spearman_rho: float
    pvalue_paired_t: float
    pvalue_wilcoxon: float

    def to_log_string(self) -> str:
        return (
            f"n={self.n} | "
            f"μ_orig={self.mean_orig:.4f} ± {self.std_orig:.4f} | "
            f"μ_adj={self.mean_adj:.4f} ± {self.std_adj:.4f} | "
            f"Pearson r={self.pearson_r:.4f} | "
            f"Spearman ρ={self.spearman_rho:.4f} | "
            f"p_t={self.pvalue_paired_t:.3e} | "
            f"p_wilcoxon={self.pvalue_wilcoxon:.3e}"
        )


# -----------------------------------------------------------------------------
# Core analysis routine
# -----------------------------------------------------------------------------

def analyze_uncertainty(
        annotations: Sequence["Annotation"],
        *,
        show: bool = True,
        save_path: Optional[Path] = None,
) -> UncertaintyDebugStats:
    """
    Compute summary statistics and (optionally) plot original vs.\ adjusted
    uncertainties.

    Parameters
    ----------
    annotations
        Iterable of Annotation objects (must expose ``uncertainty`` and
        ``adjusted_uncertainty``).
    show
        If ``True`` call ``plt.show()`` (interactive usage).
    save_path
        If provided, save the figure at this path. Parent directories are
        created automatically.

    Returns
    -------
    UncertaintyDebugStats
        Structured metrics for downstream logging or testing.
    """
    if not annotations:
        raise ValueError("No annotations provided")

    orig = np.asarray([a.uncertainty for a in annotations], dtype=float)
    adj = np.asarray([a.adjusted_uncertainty for a in annotations], dtype=float)

    stats_out = UncertaintyDebugStats(
        n=len(orig),
        mean_orig=orig.mean(),
        std_orig=orig.std(ddof=1),
        mean_adj=adj.mean(),
        std_adj=adj.std(ddof=1),
        pearson_r=np.corrcoef(orig, adj)[0, 1],
        spearman_rho=stats.spearmanr(orig, adj, nan_policy="omit").correlation,
        pvalue_paired_t=stats.ttest_rel(orig, adj, nan_policy="omit").pvalue,
        pvalue_wilcoxon=stats.wilcoxon(orig, adj, zero_method="wilcox", correction=True).pvalue,
    )

    # ------------------------------------------------------------------ plot
    if show or save_path:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        # Histogram
        bins = "auto"  # Freedman‑Diaconis via numpy
        ax[0].hist(orig, bins=bins, alpha=0.6, label="Original")
        ax[0].hist(adj, bins=bins, alpha=0.6, label="Adjusted")
        ax[0].set_xlabel("Uncertainty")
        ax[0].set_ylabel("Frequency")
        ax[0].set_title("Distribution of uncertainties")
        ax[0].legend()

        # Scatter
        reduction = orig - adj
        sc = ax[1].scatter(orig, adj, c=reduction, cmap="viridis", alpha=0.5)
        ax[1].plot([orig.min(), orig.max()], [orig.min(), orig.max()],
                   ls="--", lw=0.7, color="grey")
        ax[1].set_xlabel("Original")
        ax[1].set_ylabel("Adjusted")
        ax[1].set_title("Original vs. adjusted")
        cb = fig.colorbar(sc, ax=ax[1], label="Δ uncertainty")

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300)
            logging.info("Uncertainty debug plot saved to %s", save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return stats_out
