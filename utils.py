# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_latency_distribution(df, name="", ax=None):
    """Histogram of collector latency (recv_ts_utc - exchange_ts_utc) for one feed.

    Expects a "latency_s" column. Uses log-spaced bins because the range spans
    ~5 orders of magnitude, and percent-of-messages on y so feeds with very
    different row counts stay comparable. Pass a shared `ax` to draw several
    feeds as panels of one figure.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4))

    lat = df["latency_s"]
    bins = np.logspace(np.log10(lat.min()), np.log10(lat.max()), 60)
    ax.hist(lat, bins=bins, color = "#2a78d6", edgecolor="none",
            weights=np.full(len(lat), 100 / len(lat)))   # % so panels compare
    ax.set_xscale("log")

    med = lat.median()
    ax.axvline(med, color="#52514e", lw=1, ls="--")
    ax.annotate(f"median {med*1000:.0f} ms", xy=(med, 0.94),
                xycoords=("data", "axes fraction"), xytext=(4, 0),
                textcoords="offset points", fontsize=9, color="#52514e", va="top")
    ax.set_title(f"{name}  (n={len(lat):,})", fontsize=11, loc="left")
    ax.set_xlabel("recv − exchange (seconds, log scale)", fontsize=9, color="#52514e")
    ax.set_ylabel("% of messages", fontsize=9, color="#52514e")
    ax.grid(axis="y", color = "#e5e5e2", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    return ax
