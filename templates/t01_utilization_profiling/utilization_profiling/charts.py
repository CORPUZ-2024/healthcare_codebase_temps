"""Static PNG charts (matplotlib, headless). Palette: validated reference categorical slots.

Style rules: one axis, thin 2px lines, recessive grid, text in ink colors (never series colors),
direct label on the line instead of a legend when there is one series.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def trend_chart(monthly, path: Path, title: str) -> Path:
    """Monthly rate (thin line + markers) and rolling-12 (thicker line); runout months left as gaps."""
    x = monthly.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(x, monthly["rate_per_1000"], color=SERIES[0], linewidth=1.2, marker="o", markersize=3, alpha=0.6)
    ax.plot(x, monthly["r12_rate_per_1000"], color=SERIES[0], linewidth=2.2)
    last = monthly["r12_rate_per_1000"].last_valid_index()
    if last is not None:
        ax.annotate("rolling 12-month", (last.to_timestamp(), monthly.loc[last, "r12_rate_per_1000"]),
                    xytext=(6, 0), textcoords="offset points", color=INK_2, fontsize=9, va="center")
    ax.set_ylim(bottom=0)
    _style(ax, title, "per 1,000 member-years")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
