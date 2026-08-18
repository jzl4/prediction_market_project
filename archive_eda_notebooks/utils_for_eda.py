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


# ---------------------------------------------------------------------------
# Single-market inspection
#
# Every books row -- snapshot AND update -- carries a full top-5 ladder on both
# sides, so a row is a complete photo of the book, not a delta. Nothing labels
# what caused a change, so the cause (add / cancel / trade) has to be inferred
# by diffing consecutive photos and lining trades up against them.
# ---------------------------------------------------------------------------

def _delta(now, before):
    """Annotation for one price level vs. the previous book row."""
    if before is None:
        return "<- NEW LEVEL"
    d = now - before
    return "" if abs(d) < 1e-9 else f"<- {d:+,.2f}"


def _describe_changes(bids, asks, p_bids, p_asks):
    """Compact one-line summary of what moved between two book rows."""
    out = []
    for name, now, before in (("bid", bids, p_bids), ("ask", asks, p_asks)):
        for price in sorted(set(now) | set(before), reverse=True):
            d = now.get(price, 0.0) - before.get(price, 0.0)
            if abs(d) > 1e-9:
                out.append(f"{name} {price:.2f} {d:+,.2f}")
    return out


def _describe_gone(bids, asks, p_bids, p_asks):
    """Levels present in the previous row that are absent from this one."""
    return ([f"bid {p:.2f}" for p in sorted(set(p_bids) - set(bids), reverse=True)]
            + [f"ask {p:.2f}" for p in sorted(set(p_asks) - set(asks), reverse=True)])


def _ladder(side):
    """One side of a book row -> {price: qty}. Handles list or numpy array."""
    return {float(lvl["price"]): float(lvl["qty"]) for lvl in side}


def get_market(native_id, books, trades):
    """Slice books and trades down to one market, sorted oldest-first.

    Returns copies, so downstream edits can't touch the originals. The row's
    position in the full frame is kept as "orig_idx" for jumping back.

    Sorts on the raw microsecond recv_ts_utc, not recv_ts_sec: a single market
    can take dozens of updates inside one second, and second-granularity would
    scramble their order.
    """
    out = []
    for df in (books, trades):
        d = df[df["native_id"] == native_id].copy()
        d["recv_ts_utc"] = pd.to_datetime(d["recv_ts_utc"], utc=True)  # idempotent
        d = d.sort_values("recv_ts_utc")
        out.append(d.reset_index().rename(columns={"index": "orig_idx"}))
    return out[0], out[1]


def show_window(b, t, start, seconds=1.0, depth=5, full=True, max_events=40):
    """Print every book row and trade for one market inside a time window.

    b, t      -- the two frames from get_market()
    start     -- anything pd.Timestamp accepts, e.g. "2026-08-05 22:57:56"
    seconds   -- window length; 1.0 gives you a one-second block
    full      -- True prints the whole ladder ("whiteboard") for each book row,
                 False prints one line per row (best bid/ask + what changed)

    Changes are measured against the previous book row for this market, even
    when that row falls before the window, so the first entry still shows a
    delta rather than appearing out of nowhere.
    """
    start = pd.Timestamp(start, tz="UTC")
    end = start + pd.Timedelta(seconds=seconds)

    in_b = b[(b["recv_ts_utc"] >= start) & (b["recv_ts_utc"] < end)]
    in_t = t[(t["recv_ts_utc"] >= start) & (t["recv_ts_utc"] < end)]

    print(f"{start:%Y-%m-%d %H:%M:%S.%f} -> {end:%H:%M:%S.%f}   "
          f"{len(in_b)} book rows, {len(in_t)} trades")
    print("=" * 68)
    if in_b.empty and in_t.empty:
        print("  (nothing in this window)")
        return

    # Interleave the two feeds into one time-ordered tape.
    events = ([(ts, "book", i) for i, ts in in_b["recv_ts_utc"].items()]
              + [(ts, "trade", i) for i, ts in in_t["recv_ts_utc"].items()])
    events.sort(key=lambda e: (e[0], e[1] == "trade"))  # book before trade on ties

    if len(events) > max_events:
        print(f"  ... {len(events)} events, showing first {max_events}. "
              f"Shorten `seconds` or pass a bigger `max_events`.\n")
        events = events[:max_events]

    for ts, kind, i in events:
        if kind == "trade":
            r = t.loc[i]
            side = "BUY " if r["aggressor_buy_flag"] else "SELL"
            print(f"  {ts:%H:%M:%S.%f}  *** TRADE  {side} {r['qty']:>10,.2f} "
                  f"@ {r['price']:.2f} ***")
            print()
            continue

        r = b.loc[i]
        prev = b.loc[i - 1] if i > 0 else None
        bids, asks = _ladder(r["bids"]), _ladder(r["asks"])
        p_bids = _ladder(prev["bids"]) if prev is not None else {}
        p_asks = _ladder(prev["asks"]) if prev is not None else {}

        head = (f"  {ts:%H:%M:%S.%f}  {r['msg_type']:<8}  "
                f"best {r['best_bid']:.2f} / {r['best_ask']:.2f}"
                f"   (row {r['orig_idx']})")

        if not full:
            ch = _describe_changes(bids, asks, p_bids, p_asks)
            print(head + ("   " + "; ".join(ch) if ch else "   [no change]"))
            continue

        print(head)
        for price in sorted(asks, reverse=True)[:depth]:
            print(f"      ASK  {price:.2f} x {asks[price]:>12,.2f}"
                  f"  {_delta(asks[price], p_asks.get(price))}")
        print(f"      {'-' * 34}  spread {r['spread']:.2f}")
        for price in sorted(bids, reverse=True)[:depth]:
            print(f"      BID  {price:.2f} x {bids[price]:>12,.2f}"
                  f"  {_delta(bids[price], p_bids.get(price))}")

        gone = _describe_gone(bids, asks, p_bids, p_asks)
        if gone:
            print(f"      dropped off the top {depth}: " + ", ".join(gone))
        print()
