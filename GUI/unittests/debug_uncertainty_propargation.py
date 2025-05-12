from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import count
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
        # NEW optional overrides  ↓↓↓
        hist_range: tuple[float, float] | None = None,
        scatter_xlim: tuple[float, float] | None = None,
        scatter_ylim: tuple[float, float] | None = None,
        diff_clim: tuple[float, float] | None = None,
) -> UncertaintyDebugStats:
    """
    Compute summary statistics and (optionally) plot original vs. adjusted
    uncertainties.  If orig == adj everywhere, we assign perfect correlations
    (r = ρ = 1) and p-values = 1 to indicate no evidence against the null.
    """
    if not annotations:
        raise ValueError("No annotations provided")

    orig = np.asarray([a.uncertainty for a in annotations], dtype=float)
    adj = np.asarray([a.adjusted_uncertainty for a in annotations], dtype=float)
    diff = orig - adj

    if np.allclose(diff, 0):
        pearson_r = 1.0
        spearman_rho = 1.0
        pvalue_paired_t = 1.0
        pvalue_wilcoxon = 1.0
    else:
        pearson_r = np.corrcoef(orig, adj)[0, 1]
        spearman_rho = stats.spearmanr(orig, adj, nan_policy="omit").correlation
        pvalue_paired_t = stats.ttest_rel(orig, adj, nan_policy="omit").pvalue
        pvalue_wilcoxon = stats.wilcoxon(orig, adj,
                                         zero_method="wilcox",
                                         correction=True).pvalue

    stats_out = UncertaintyDebugStats(
        n=len(orig),
        mean_orig=orig.mean(),
        std_orig=orig.std(ddof=1),
        mean_adj=adj.mean(),
        std_adj=adj.std(ddof=1),
        pearson_r=pearson_r,
        spearman_rho=spearman_rho,
        pvalue_paired_t=pvalue_paired_t,
        pvalue_wilcoxon=pvalue_wilcoxon,
    )

    # ------------------------------------------------------------------ plot
    if show or save_path:
        fig, ax = plt.subplots(2, 1, figsize=(6, 8), constrained_layout=True)

        # Histogram
        bins = "auto"
        ax[0].hist(orig, bins=bins, alpha=0.6, label="Original", range=hist_range)
        ax[0].hist(adj, bins=bins, alpha=0.6, label="Adjusted", range=hist_range)
        if hist_range is not None:
            ax[0].set_xlim(hist_range)

        # Scatter
        sc = ax[1].scatter(orig, adj, c=diff, cmap="viridis", alpha=0.5,
                           vmin=None if diff_clim is None else diff_clim[0],
                           vmax=None if diff_clim is None else diff_clim[1])
        if scatter_xlim is not None:
            ax[1].set_xlim(scatter_xlim)
        if scatter_ylim is not None:
            ax[1].set_ylim(scatter_ylim)
        ax[1].set_xlabel("Original")
        ax[1].set_ylabel("Adjusted")
        ax[1].set_title("Original vs. adjusted")
        fig.colorbar(sc, ax=ax[1], label="Δ uncertainty")

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


class ProgressFilmer:
    FRAME_EVERY_N = 2
    _counter = count(0)

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # -------- running extrema --------
        self.xmin = np.inf
        self.xmax = -np.inf
        self.ymin = np.inf
        self.ymax = -np.inf
        self.dmin = np.inf  # diff = orig-adj
        self.dmax = -np.inf

    # --------------------------------------------------- public hook
    def maybe_record_frame(self, annotations: Sequence["Annotation"]) -> None:
        i = next(self._counter)
        if i % self.FRAME_EVERY_N:
            return

        # --- update global extrema first ---------------------------
        orig = np.asarray([a.uncertainty for a in annotations], dtype=float)
        adj = np.asarray([a.adjusted_uncertainty for a in annotations], dtype=float)
        diff = orig - adj

        self.xmin = min(self.xmin, orig.min())
        self.xmax = max(self.xmax, orig.max())
        self.ymin = min(self.ymin, adj.min())
        self.ymax = max(self.ymax, adj.max())
        self.dmin = min(self.dmin, diff.min())
        self.dmax = max(self.dmax, diff.max())

        # --- hand off to the plotting routine ----------------------
        analyze_uncertainty(
            annotations,
            show=False,
            save_path=self.out_dir / f"frame_{i:05d}.png",
            hist_range=(self.xmin, self.xmax),
            scatter_xlim=(self.xmin, self.xmax),
            scatter_ylim=(self.ymin, self.ymax),
            diff_clim=(self.dmin, self.dmax),
        )
        plt.close('all')
