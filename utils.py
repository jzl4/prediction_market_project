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


# ---------------------------------------------------------------------------
# Q1 -- market efficiency metrics
#
# Each function takes the rows for ONE market and returns one number, so they
# can be applied across all 33 markets and assembled into a table.
#
# TAKER_FEE_RATE is from the instructions: taking costs p*(1-p)*0.07 per
# contract, where p is the price you actually pay. Making is free.
# ---------------------------------------------------------------------------

TAKER_FEE_RATE = 0.07


def _one_market(b):
    """Guard + sort. Every metric below assumes a single market in time order.

    Passing a multi-market frame would silently corrupt the dwell times (the
    gap between the last row of one market and the first row of the next is
    not a dwell), so refuse it outright rather than return a wrong number.
    """
    n = b["native_id"].nunique()
    if n != 1:
        raise ValueError(f"expected rows for exactly 1 market, got {n}")
    return b.sort_values("recv_ts_utc")


def _dwell_seconds(b):
    """How long each row's quote stood before the next message replaced it.

    The feed only sends a message when something changes, so rows are events,
    not clock ticks. This is the weight that converts "average per message"
    into "average per second".

    The last row has no successor, so its dwell is NaN and it drops out of the
    weighted averages below. That loses one quote per market out of hundreds.
    """
    return b["recv_ts_utc"].diff().shift(-1).dt.total_seconds()


def _weighted_mean(values, weights):
    """Mean of `values` weighted by `weights`, ignoring rows where either is NaN."""
    ok = values.notna() & weights.notna()
    if not ok.any():
        return float("nan")
    return float((values[ok] * weights[ok]).sum() / weights[ok].sum())


def _ladder_total_qty(side):
    """Sum the quantities across one side's ladder (all 5 levels)."""
    return sum(float(lvl["qty"]) for lvl in side)


def time_weighted_relative_spread(b):
    """Average spread as a fraction of mid price, weighted by time.

    Relative, not raw cents, because the tick is 1 cent and the book sits at
    1 tick in ~87% of rows -- raw cents ties nearly all 33 markets at 0.01.
    Dividing by mid separates them: 1 cent is ~1% of a contract trading at
    0.96 but ~19% of one trading at 0.105.

    Returns a fraction (0.01 = 1% of mid). Multiply by 100 to report percent.
    """
    b = _one_market(b)
    mid = (b["best_bid"] + b["best_ask"]) / 2
    return _weighted_mean(b["spread"] / mid, _dwell_seconds(b))


def median_total_quantity_top_5(b):
    """Median resting depth in contracts: top 5 bids + top 5 asks, per row.

    Contracts, not dollars. Every contract here settles at exactly $0 or $1,
    so face value is identical across all 33 markets and quantity is already
    the notional-equivalent -- unlike stocks, where you multiply by price
    because face values differ.

    Note this one is event-weighted, not time-weighted like the spread
    metrics. A median only depends on ordering, so it is far less distorted by
    busy periods producing extra messages than a mean would be. Kept as a
    median for simplicity; switch to a dwell-weighted mean if it ever matters.
    """
    b = _one_market(b)
    depth = b["bids"].map(_ladder_total_qty) + b["asks"].map(_ladder_total_qty)
    return float(depth.median())


def total_volume_traded(t):
    """Total contracts traded in this market over the whole window.

    Takes the TRADES frame, not the books frame. Returns 0.0 for a market that
    never traded -- 2 of the 33 are quoted but never print.
    """
    if len(t) == 0:
        return 0.0
    t = _one_market(t)
    return float(t["qty"].sum())


def time_weighted_cost_to_cross(b):
    """What it actually costs a taker to buy, as a fraction of mid, time-weighted.

    Two components, both per contract:
      1. half-spread -- lifting the ask means paying (best_ask - mid)
      2. taker fee   -- p*(1-p)*0.07, where p is the price paid, so best_ask.
         This matches the instructions' example: at 0.49/0.50, taking costs
         0.50*0.50*0.07 on top of the 0.50.

    Worth computing separately from the spread because the fee peaks at
    p = 0.50 and vanishes at the extremes, so it reorders the ranking: at
    p = 0.50 the fee is ~1.75 cents against a 0.5 cent half-spread.

    Priced from the buy side for a single comparable number. The sell side is
    the mirror image and gives the same ranking.
    """
    b = _one_market(b)
    mid = (b["best_bid"] + b["best_ask"]) / 2
    half_spread = b["best_ask"] - mid
    fee = b["best_ask"] * (1 - b["best_ask"]) * TAKER_FEE_RATE
    return _weighted_mean((half_spread + fee) / mid, _dwell_seconds(b))
