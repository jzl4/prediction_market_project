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


# ---------------------------------------------------------------------------
# Q4b -- rolling out the signal under a $1,000 loss tolerance
#
# STEP 1: the naive simulation. Constant bet size, even-money trades (p = 0.50),
# no fees. Win $X or lose $X on each trade, 55/45.
#
# p = 0.50 is not an arbitrary choice: the prompt's phrase "10% edge over a 50%
# breakeven" only makes sense at that price. Expected profit per contract is
# (w - p), so a 55% win rate is an edge of 5 cents at p = 0.50, but would be
# MINUS 25 cents at p = 0.80. Win rate means nothing without the price.
#
# Later steps (real price paths, dynamic sizing, fees) are not implemented yet.
# ---------------------------------------------------------------------------

def simulate_naive_random_walk(bet_size, n_trades=32_000, win_prob=0.55,
                               loss_limit=-1000.0, n_paths=10_000,
                               batch_size=500, seed=0):
    """Monte Carlo of a constant-size, even-money betting sequence.

    bet_size   -- dollars won or lost per trade
    n_trades   -- steps per path. 32,000 = 500 trades/day x 64 days. Only the
                  total matters, not how it splits into days: with a constant
                  size, nothing in the walk depends on the calendar.
    loss_limit -- absolute floor on CUMULATIVE P&L, not a drawdown from peak.
                  Those are different rules and give different answers.
    n_paths    -- independent simulated futures.
    batch_size -- paths simulated at once. n_paths x n_trades floats would be
                  ~2.5 GB at the defaults, so the work is done in batches.

    Ruin is recorded but does not stop the path -- the walk runs the full
    n_trades either way. That keeps "did it ever touch the floor" separate from
    "where did it end up", so both can be read off the same run.

    Returns a dict of summary statistics, one row's worth per call.
    """
    rng = np.random.default_rng(seed)

    ruined      = np.zeros(n_paths, dtype=bool)     # ever touched the floor?
    first_ruin  = np.full(n_paths, -1, dtype=np.int64)   # trade number if so
    final_pnl   = np.zeros(n_paths)
    worst_pnl   = np.zeros(n_paths)                 # lowest point along the way

    for start in range(0, n_paths, batch_size):
        n = min(batch_size, n_paths - start)

        # +bet_size with probability win_prob, -bet_size otherwise
        wins  = rng.random((n, n_trades)) < win_prob
        steps = np.where(wins, bet_size, -bet_size)
        pnl   = np.cumsum(steps, axis=1)            # running P&L along each path

        below = pnl <= loss_limit
        hit   = below.any(axis=1)
        # argmax on a boolean row gives the first True; meaningless if none, so mask it
        first = np.where(hit, below.argmax(axis=1), -1)

        sl = slice(start, start + n)
        ruined[sl]     = hit
        first_ruin[sl] = first
        final_pnl[sl]  = pnl[:, -1]
        worst_pnl[sl]  = pnl.min(axis=1)

    survivors = ~ruined
    return {
        "bet_size":        bet_size,
        "prob_ruin":       float(ruined.mean()),
        # when ruin happens, how early? The prompt's real question is whether we
        # survive the opening stretch, so the timing matters as much as the odds.
        "median_ruin_trade": float(np.median(first_ruin[ruined])) if ruined.any() else np.nan,
        "median_final_pnl":  float(np.median(final_pnl)),
        "median_final_pnl_survivors": float(np.median(final_pnl[survivors])) if survivors.any() else np.nan,
        "median_worst_pnl":  float(np.median(worst_pnl)),
        "p05_worst_pnl":     float(np.percentile(worst_pnl, 5)),
    }


def ruin_probability_closed_form(bet_size, win_prob=0.55, loss_limit=-1000.0):
    """Gambler's ruin: chance of ever touching the floor, as a check on the sim.

    For a +/-X bet with win probability w > 0.5 and a floor k bets away, the
    probability of ever reaching that floor is ((1-w)/w) ** k.

    This assumes infinitely many trades, so it is a slight OVER-estimate of what
    a 32,000-trade simulation should produce -- the sim has a finite number of
    chances to get there. Expect the simulation to land at or just below this.

    Used as a unit test, not as the answer: if the Monte Carlo does not land
    near these numbers, the Monte Carlo has a bug.
    """
    k = abs(loss_limit) / bet_size          # how many losing bets to the floor
    return float(((1 - win_prob) / win_prob) ** k)


def simulate_pnl_paths(bet_size, n_trades=32_000, win_prob=0.55,
                       n_paths=40, seed=0):
    """The same walk as simulate_naive_random_walk, but keeping the whole path.

    Returns an (n_paths, n_trades) array of cumulative P&L, so the wiggle can be
    plotted rather than just summarised. Keep n_paths small -- this holds every
    step in memory, unlike the batched summary version.
    """
    rng = np.random.default_rng(seed)
    steps = np.where(rng.random((n_paths, n_trades)) < win_prob, bet_size, -bet_size)
    return np.cumsum(steps, axis=1)


def plot_pnl_paths(paths, loss_limit=-1000.0, zoom=1000, max_points=2000, axes=None):
    """Draw simulated P&L paths: the whole run, and a zoom on the opening stretch.

    Two panels are necessary rather than decorative. With a 10% edge over 32,000
    trades the paths finish in the hundreds of thousands, while the floor we care
    about is -$1,000 -- less than 1% of the y-range. On a single full-scale axis
    the floor and the early wiggle are invisible. The right panel rescales to the
    first `zoom` trades, which is where ruin actually happens.

    Paths that ever touch the floor are drawn in red on top of the survivors, so
    the failures are visible even when they are a small minority.

    paths      -- (n_paths, n_trades) array from simulate_pnl_paths()
    loss_limit -- the floor to draw and to classify paths against
    zoom       -- how many trades the right-hand panel covers
    max_points -- points drawn per line per panel. Lines are thinned to this so
                  the figure stays light enough for an inline notebook renderer.
                  Only affects drawing: whether a path is classed as ruined is
                  always decided on the full, unthinned path.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    n_paths, n_trades = paths.shape
    x = np.arange(1, n_trades + 1)
    ruined = paths.min(axis=1) <= loss_limit

    SURVIVE, RUIN, RULE = "#2a78d6", "#d1442f", "#52514e"

    for ax, upto, title, label_floor in (
        (axes[0], n_trades, f"All {n_trades:,} trades", False),
        (axes[1], min(zoom, n_trades), f"First {min(zoom, n_trades):,} trades", True),
    ):
        # Drawing every point would put n_paths x n_trades vertices on the canvas
        # -- 1.28M at the defaults, which hangs an inline notebook renderer. At
        # full scale a 32,000-point line is far narrower than a pixel per step, so
        # thinning to ~max_points loses nothing visible. The zoomed panel is short
        # enough that stride comes out as 1 and it is drawn at full resolution.
        stride = max(1, upto // max_points)
        xs = x[:upto:stride]

        # survivors first, failures on top so they are never hidden underneath
        for i in np.flatnonzero(~ruined):
            ax.plot(xs, paths[i, :upto:stride], color=SURVIVE, lw=0.7, alpha=0.45)
        for i in np.flatnonzero(ruined):
            ax.plot(xs, paths[i, :upto:stride], color=RUIN, lw=0.9, alpha=0.9)

        ax.axhline(0, color=RULE, lw=1)
        ax.axhline(loss_limit, color=RUIN, lw=1.2, ls="--")
        # Only label the floor on the zoomed panel. At full scale -$1,000 and $0
        # are the same line to the eye, so a label there would mislead.
        if label_floor:
            ax.annotate(f"loss tolerance  ${loss_limit:,.0f}", xy=(0.99, loss_limit),
                        xycoords=("axes fraction", "data"), xytext=(0, 5),
                        textcoords="offset points", fontsize=8.5, color=RUIN,
                        ha="right", annotation_clip=False)

        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("trade number", fontsize=9, color=RULE)
        ax.set_ylabel("cumulative P&L ($)", fontsize=9, color=RULE)
        ax.grid(color="#e5e5e2", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # The zoom panel is the whole point, so make sure the floor is inside its
    # y-range even when every path happens to stay well above it.
    lo, hi = axes[1].get_ylim()
    axes[1].set_ylim(min(lo, loss_limit * 1.35), hi)

    axes[0].legend(handles=[
        plt.Line2D([], [], color=SURVIVE, lw=1.4,
                   label=f"survived ({(~ruined).sum()}/{n_paths})"),
        plt.Line2D([], [], color=RUIN, lw=1.4,
                   label=f"hit the floor ({ruined.sum()}/{n_paths})"),
    ], fontsize=8.5, frameon=False, loc="upper left")

    return axes


# ---------------------------------------------------------------------------
# Q4b step 2 -- trading the price move, not holding to settlement
#
# The signal calls the next move up or down with `signal_accuracy`. We enter at
# p and exit once the price has moved `delta` either way. This is the reading
# the prompt supports: 500 UNCORRELATED trades per day is impossible if we hold
# to settlement, because every position on one game shares a single outcome.
#
# Four scenarios, since the exit price -- and therefore the exit fee -- depends
# on both which way we traded and whether the signal was right:
#
#     buy,  right   ->  exit at p + delta      0.5 * accuracy
#     sell, right   ->  exit at p - delta      0.5 * accuracy
#     buy,  wrong   ->  exit at p - delta      0.5 * (1 - accuracy)
#     sell, wrong   ->  exit at p + delta      0.5 * (1 - accuracy)
#
# P(buy) is 0.5 because signal-says-up = 0.5*acc + 0.5*(1-acc) = 0.5.
# ---------------------------------------------------------------------------

def _fill_price(mid, side, spread, taking):
    """Price actually filled at, given the mid and whether we crossed.

    side = +1 to buy, -1 to sell.

    Taking means crossing: a buyer lifts the ask at mid + spread/2, a seller
    hits the bid at mid - spread/2. Either way half the spread is paid away.

    Making means resting: a buyer's bid sits at mid - spread/2 and gets hit, a
    seller's offer sits at mid + spread/2 and gets lifted. Half the spread is
    EARNED rather than paid. That sign flip is the whole reason passive
    execution is the only version of this strategy that clears its costs.
    """
    return mid + (spread / 2.0) * side * (1.0 if taking else -1.0)


def simulate_signal_trades(p=0.50, delta=0.10, spread=0.01, signal_accuracy=0.55,
                           n_contracts=100, n_trades=32_000,
                           take_on_entry=True, take_on_exit=True, seed=0):
    """Monte Carlo of a directional signal traded in and out of the book.

    p               -- entry price (mid). The market's level.
    delta            -- how far the price moves before we exit.
    spread           -- full bid/ask spread. Crossing to exit costs this once
                        per round trip: a long sells the bid, a short buys the
                        ask, so the exit price is worse by `spread` either way.
    signal_accuracy  -- how often the signal calls the direction right.
    n_contracts      -- position size, so P&L is in dollars not cents/contract.
    take_on_entry /
    take_on_exit     -- whether each leg crosses (pays p*(1-p)*0.07) or rests
                        passively (free). Making is the only way this strategy
                        clears its costs at realistic move sizes.

    Prices are clipped to [0, 1]: at p = 0.95 with delta = 0.10 the upside
    truncates at 1.0 (a 0.05 win) while the downside stays a full 0.10, so
    extreme markets are structurally worse for this strategy.

    Returns a dict including the per-trade P&L and its cumulative sum, so the
    path can be plotted directly.
    """
    rng = np.random.default_rng(seed)

    # +1 = we bought, -1 = we sold. Independent of whether the signal is right.
    direction = np.where(rng.random(n_trades) < 0.5, 1.0, -1.0)
    right     = rng.random(n_trades) < signal_accuracy

    # Which way the market actually went. Buy+right and sell+wrong mean it rose.
    move = direction * np.where(right, 1.0, -1.0)
    exit_mid = np.clip(p + delta * move, 0.0, 1.0)

    entry_px = np.clip(_fill_price(p, direction, spread, take_on_entry), 0.0, 1.0)
    # Exiting reverses the trade: closing a long means selling, so the same
    # helper with the same `direction` gives the right sign on the other leg.
    exit_px  = np.clip(_fill_price(exit_mid, -direction, spread, take_on_exit), 0.0, 1.0)

    gross = direction * (exit_px - entry_px)           # per contract, in dollars

    fee_in  = entry_px * (1 - entry_px) * TAKER_FEE_RATE if take_on_entry else 0.0
    fee_out = exit_px * (1 - exit_px) * TAKER_FEE_RATE if take_on_exit else 0.0

    per_contract = gross - fee_in - fee_out
    per_trade    = per_contract * n_contracts

    return {
        "p": p, "delta": delta, "spread": spread,
        "signal_accuracy": signal_accuracy, "n_contracts": n_contracts,
        "take_on_entry": take_on_entry, "take_on_exit": take_on_exit,
        "ev_per_contract": float(per_contract.mean()),
        "ev_per_contract_exact": signal_trade_ev(
            p, delta, spread, signal_accuracy, take_on_entry, take_on_exit),
        "ev_per_trade": float(per_trade.mean()),
        "total_pnl": float(per_trade.sum()),
        "per_trade": per_trade,
        "pnl_path": np.cumsum(per_trade),
    }


def signal_trade_ev(p=0.50, delta=0.10, spread=0.01, signal_accuracy=0.55,
                    take_on_entry=True, take_on_exit=True):
    """Exact expected P&L per contract, by enumerating the four scenarios.

    No simulation involved -- this is what the Monte Carlo above should converge
    to, so it doubles as a check on it.
    """
    acc = signal_accuracy
    scenarios = [                       # (direction, signal right, probability)
        ( 1.0, True,  0.5 * acc),       # buy,  right
        (-1.0, True,  0.5 * acc),       # sell, right
        ( 1.0, False, 0.5 * (1 - acc)), # buy,  wrong
        (-1.0, False, 0.5 * (1 - acc)), # sell, wrong
    ]

    clip = lambda x: min(max(x, 0.0), 1.0)

    ev = 0.0
    for direction, is_right, prob in scenarios:
        move     = direction * (1.0 if is_right else -1.0)
        exit_mid = clip(p + delta * move)

        entry_px = clip(_fill_price(p, direction, spread, take_on_entry))
        exit_px  = clip(_fill_price(exit_mid, -direction, spread, take_on_exit))

        gross   = direction * (exit_px - entry_px)
        fee_in  = entry_px * (1 - entry_px) * TAKER_FEE_RATE if take_on_entry else 0.0
        fee_out = exit_px * (1 - exit_px) * TAKER_FEE_RATE if take_on_exit else 0.0
        ev     += prob * (gross - fee_in - fee_out)

    return float(ev)


# ===========================================================================
# Q1 METRICS -- for q1_market_efficiency_new.ipynb
#
# These three functions belong to the NEW Q1 notebook and follow the
# definitions written there. They are deliberately separate from the older
# Q1 block above (time_weighted_relative_spread, time_weighted_cost_to_cross,
# median_total_quantity_top_5, total_volume_traded), which belongs to
# q1_market_efficiency_claude.ipynb and is kept only so that notebook still
# runs. Do not mix the two sets: they normalise differently and will not agree.
#
# What changed between the two:
#   - spread is now divided by p*(1-p), not by the mid. Dividing by the mid
#     silently assumes the trader is buying "yes"; s/(p*(1-p)) is the sum of
#     the cost in both directions and does not change when p and (1-p) swap.
#   - the aggregator across time is the MEDIAN for both spread and depth,
#     not a dwell-weighted mean.
# ===========================================================================


def median_relative_trading_cost(b):
    """(1) Median cost of crossing the spread, as a fraction of capital deployed.

    Per row of the books frame -- snapshot and update alike, since both carry a
    complete photo of the book -- compute

        s / (p * (1 - p))

    where s = best_ask - best_bid and p = the mid, (best_bid + best_ask) / 2.

    Why that denominator rather than the mid: buying "yes" deploys p of capital
    and costs s/p; selling "yes" deploys (1 - p) and costs s/(1 - p). Summing
    the two directions gives s/p + s/(1-p) = s/(p*(1-p)), which is unchanged
    when p and (1-p) swap places. So it measures the market, not whichever
    direction we happened to pick.

    Takes the MEDIAN across all rows in the window, per the notebook's
    instruction. A median depends only on ordering, so unlike a mean it is not
    dragged around by the handful of rows where the book was briefly wide.

    Returns a fraction: 0.11 means 11% of capital deployed.
    """
    b = _one_market(b)

    # p and s straight off the top of book, one value per photo of the ladder
    mid = (b["best_bid"] + b["best_ask"]) / 2
    spread = b["best_ask"] - b["best_bid"]

    # cost in both directions at once; mid is never 0 or 1 in this data, so
    # the denominator cannot vanish
    cost = spread / (mid * (1 - mid))

    return float(cost.median())


def median_order_book_depth(b):
    """(2) Median resting size across both sides of the book.

    Per row, add up the qty at every level the feed gives us -- up to 5 bids
    and up to 5 asks -- then take the median of that total across the window.

    Short ladders are counted as-is, not padded or dropped. The ask side has
    fewer than 5 levels in about 4.5% of rows, and a book quoting only 2 or 3
    levels really is thinner than one quoting 5, so letting the sum come out
    smaller is the correct behaviour rather than a gap to patch.

    Units are whatever the feed's qty field is denominated in. It is almost
    certainly not a plain contract count -- only 41.7% of trade quantities are
    integers, while qty * 100 is an integer 100% of the time -- but the unit is
    consistent across all 33 markets, so it ranks them correctly either way.
    """
    b = _one_market(b)

    # _ladder_total_qty sums whatever levels are present on one side
    depth = b["bids"].map(_ladder_total_qty) + b["asks"].map(_ladder_total_qty)

    return float(depth.median())


def total_contracts_traded(t):
    """(3) Total quantity traded in this market over the whole window.

    Takes the TRADES frame, not the books frame. Every trade's qty is added to
    a running total; this is a flow over the session, not a state, so there is
    no mean-vs-median choice to make -- it is simply a sum.

    Returns 0.0 for a market that is quoted but never prints. Two of the 33 are
    in that position, so this cannot be left to raise.
    """
    if len(t) == 0:
        return 0.0

    t = _one_market(t)
    return float(t["qty"].sum())


# ===========================================================================
# Q1 (4) ARBITRAGE / INTERNAL CONSISTENCY -- for q1_market_efficiency_new.ipynb
#
# Every contract here is a survival probability S(n) = P(runs >= n), so a set
# of known inequalities must hold between them at any single instant. We walk
# the book feed in time order and test every one we can observe.
#
# The hard part is not the inequalities, it is TIME. Adjacent strikes never
# share a timestamp (F5TOTAL-5 and F5TOTAL-6 have 0 timestamps in common out
# of 954 and 876), so any comparison pairs a fresh quote against an older one.
# And we cannot assume an old quote is still valid: 66% of trades land inside
# a gap of >5s in the book feed, which proves the feed does NOT report every
# change. So each cached quote carries its age and is discarded once it goes
# past a freshness threshold.
#
# Prices compared are MIDS. That is a screen, not a compromise: an executable
# arbitrage needs bid(m) > ask(n) for n < m, and since ask(n) > mid(n) and
# mid(m) > bid(m), that implies mid(m) > mid(n). Every executable violation is
# therefore also a mid violation, so finding none on mids PROVES none are
# executable and no fee/spread analysis is needed.
# ===========================================================================

GAME_STAMP = "-26AUG051940PITMIL"


def market_short_name(native_id):
    """Strip the shared game stamp so market names are readable."""
    return native_id.replace(GAME_STAMP, "")


def parse_chain_and_strike(short_name):
    """Split a market name into (chain, strike).

    The strike suffix IS the survival strike: KXMLBTOTAL-8 carries line 7.5,
    i.e. "over 7.5 runs", i.e. P(runs >= 8). Verified against the `line`
    column for all 33 markets -- line is always (strike - 0.5).

    The four ladders are returned as distinct chains. TEAMTOTAL in particular
    must split into MIL and PIT: those are different random variables (each
    team's own runs), so there is NO ordering constraint between them and
    comparing across them would manufacture fake violations.

    RFI is a single binary with no ladder, so it gets strike 1 (its line is
    0.5, "at least 1 run in the 1st inning") and a chain of its own. It never
    participates in a monotonicity test -- there is nothing to be monotone
    against -- but it does take part in the nesting test below.
    """
    if short_name.startswith("KXMLBRFI"):
        return "RFI", 1
    if short_name.startswith("KXMLBTEAMTOTAL-MIL"):
        return "TEAMTOTAL-MIL", int(short_name.rsplit("MIL", 1)[1])
    if short_name.startswith("KXMLBTEAMTOTAL-PIT"):
        return "TEAMTOTAL-PIT", int(short_name.rsplit("PIT", 1)[1])
    if short_name.startswith("KXMLBF5TOTAL-"):
        return "F5TOTAL", int(short_name.rsplit("-", 1)[1])
    if short_name.startswith("KXMLBTOTAL-"):
        return "TOTAL", int(short_name.rsplit("-", 1)[1])
    raise ValueError(f"unrecognised market name: {short_name}")


def market_label(short_name):
    """Short readable label for the logs, e.g. 'S_TOTAL(7)', 'S_MIL(5)'."""
    chain, strike = parse_chain_and_strike(short_name)
    pretty = {"F5TOTAL": "F5", "TEAMTOTAL-MIL": "MIL",
              "TEAMTOTAL-PIT": "PIT", "TOTAL": "TOTAL", "RFI": "RFI"}[chain]
    return f"S_{pretty}({strike})"


def build_constraint_pairs(short_names):
    """All ordered pairs (a, b) for which the constraint says S_a <= S_b.

    Returns {(a, b): rule_description}. Looking up a pair in BOTH orders tells
    you whether two markets are comparable at all, and if so which direction
    the inequality runs. Markets with no entry either way -- MIL against PIT,
    say -- are simply skipped by the engine, with no special-case code.

    The four families of constraint:

    1. WITHIN-CHAIN MONOTONICITY. S(n) = P(runs >= n) is a survival function,
       so it is non-increasing in n. Every game with 5+ runs is also a game
       with 4+ runs. Checked for ALL pairs of strikes in a chain, not just
       adjacent ones: because of the freshness filter we often will not have
       two adjacent strikes fresh at the same moment, and restricting to
       adjacent pairs would throw most comparisons away for nothing.

    2. RFI NESTS INSIDE F5. Inning 1 is contained in innings 1-5, so runs in
       the 1st <= runs in the first five, on every single game. Testable at
       the one strike they share, n = 1.

    3. F5 NESTS INSIDE THE FULL GAME. Innings 1-5 are contained in innings
       1-9. Shared strikes n = 2..7.

    4. EITHER TEAM NESTS INSIDE THE GAME TOTAL. Runs are non-negative, so if
       Milwaukee alone scores 5 the game total is automatically >= 5. Shared
       strikes n = 2..8. Note this is the weak `max` bound and NOT the
       convolution: the distribution of a sum only equals the convolution of
       the marginals if the two are independent, and two teams in the same
       game are not.
    """
    parsed = {m: parse_chain_and_strike(m) for m in short_names}
    by_chain = {}
    for m, (chain, strike) in parsed.items():
        by_chain.setdefault(chain, {})[strike] = m

    pairs = {}

    # --- 1. within-chain monotonicity, all pairs of strikes -----------------
    for chain, strikes in by_chain.items():
        if chain == "RFI":
            continue                      # single contract, nothing to compare
        ks = sorted(strikes)
        for i, n in enumerate(ks):
            for m in ks[i + 1:]:          # m > n, so S(m) must be <= S(n)
                pairs[(strikes[m], strikes[n])] = (
                    f"{chain} monotonicity: S({m}) <= S({n})")

    # --- 2/3/4. nesting across chains, at strikes the two chains share ------
    def add_nesting(inner_chain, outer_chain, why):
        inner, outer = by_chain.get(inner_chain, {}), by_chain.get(outer_chain, {})
        for strike in sorted(set(inner) & set(outer)):
            pairs[(inner[strike], outer[strike])] = f"{why} at strike {strike}"

    add_nesting("RFI",           "F5TOTAL", "inning 1 nests in innings 1-5")
    add_nesting("F5TOTAL",       "TOTAL",   "innings 1-5 nest in the full game")
    add_nesting("TEAMTOTAL-MIL", "TOTAL",   "MIL runs nest in the game total")
    add_nesting("TEAMTOTAL-PIT", "TOTAL",   "PIT runs nest in the game total")

    return pairs


def describe_violation(v):
    """Render one violation as a human-readable block.

    Shows both quotes with their own timestamps, how far apart they were, and
    which rule broke and by how much -- so anomalies can be read and sanity
    checked rather than only counted.
    """
    fmt = lambda ts: pd.Timestamp(ts).strftime("%H:%M:%S.%f")[:-3]
    return (f"{fmt(v['t_earlier'])}  {v['label_earlier']:<12} mid {v['mid_earlier']:.3f}\n"
            f"{fmt(v['t_later'])}  {v['label_later']:<12} mid {v['mid_later']:.3f}"
            f"   ({v['gap_seconds']:.3f} s later)\n"
            f"-> violates {v['rule']} by {v['violation_cents']:.1f}c")


def check_arbitrage_violations(books, freshness_seconds=1.0):
    """Walk the book feed in time order and test every observable constraint.

    Algorithm, one pass in chronological order:
      - keep a cache holding the LATEST quote per market (dict keyed by market,
        so a market updating three times inside the window leaves one entry,
        not three)
      - on each arriving book, first evict every cached quote older than
        `freshness_seconds`
      - then compare the arriving quote against each surviving cached quote
        that has a known constraint direction, and record any violation
      - finally insert the arriving quote into the cache

    One row per DETECTION, not per episode. We cannot know when a violation
    ended -- the pair may simply go stale before we see it again -- and it does
    not matter: two mutually inconsistent prices observed within
    `freshness_seconds` of each other is the data point. No censoring logic.

    Returns a dict with the violations frame, the readable log, and the
    comparison count that forms the denominator.
    """
    b = books.copy()
    b["recv_ts_utc"] = pd.to_datetime(b["recv_ts_utc"], utc=True)
    b["market"] = b["native_id"].map(market_short_name)
    b = b.sort_values("recv_ts_utc")

    constraints = build_constraint_pairs(sorted(b["market"].unique()))
    labels = {m: market_label(m) for m in b["market"].unique()}

    cache = {}                 # market -> (timestamp, mid)
    violations, n_comparisons = [], 0

    # Tightest slack ever seen on each rule, in cents. Slack = (S_b - S_a),
    # the room left before the constraint would break. A null result is only
    # meaningful alongside this: "no violations" is unimpressive if the strikes
    # were never anywhere near each other.
    tightest = {}
    window = pd.Timedelta(seconds=freshness_seconds)

    for row in b.itertuples(index=False):
        now, mkt = row.recv_ts_utc, row.market
        mid_now = (row.best_bid + row.best_ask) / 2

        # evict anything that has gone stale as of this instant
        for other in [k for k, (ts, _) in cache.items() if now - ts > window]:
            del cache[other]

        for other, (ts_other, mid_other) in cache.items():
            if other == mkt:
                continue

            # is this pair comparable, and if so in which direction?
            # (a, b) in constraints means the rule requires S_a <= S_b
            if (mkt, other) in constraints:
                a, b_, mid_a, mid_b = mkt, other, mid_now, mid_other
            elif (other, mkt) in constraints:
                a, b_, mid_a, mid_b = other, mkt, mid_other, mid_now
            else:
                continue                    # no known relationship -- skip

            n_comparisons += 1

            slack_cents = 100 * (mid_b - mid_a)
            rule = constraints[(a, b_)]
            if rule not in tightest or slack_cents < tightest[rule]:
                tightest[rule] = slack_cents

            if mid_a <= mid_b:
                continue                    # constraint holds

            # violation: record which quote arrived first for the log
            earlier_is_cached = ts_other < now
            violations.append({
                "t_earlier":       ts_other if earlier_is_cached else now,
                "t_later":         now if earlier_is_cached else ts_other,
                "label_earlier":   labels[other] if earlier_is_cached else labels[mkt],
                "label_later":     labels[mkt] if earlier_is_cached else labels[other],
                "mid_earlier":     mid_other if earlier_is_cached else mid_now,
                "mid_later":       mid_now if earlier_is_cached else mid_other,
                "gap_seconds":     abs((now - ts_other).total_seconds()),
                "market_a":        a,
                "market_b":        b_,
                "rule":            constraints[(a, b_)],
                "mid_a":           mid_a,
                "mid_b":           mid_b,
                "violation_cents": 100 * (mid_a - mid_b),
                "threshold_s":     freshness_seconds,
            })

        cache[mkt] = (now, mid_now)

    v = pd.DataFrame(violations)
    slack = (pd.Series(tightest, name="tightest_slack_cents")
               .sort_values().rename_axis("rule").to_frame())
    return {
        "violations":    v,
        "log":           [describe_violation(x) for x in violations],
        "slack":         slack,
        "n_comparisons": n_comparisons,
        "n_violations":  len(violations),
        "n_constraints": len(constraints),
        "threshold_s":   freshness_seconds,
    }


def arbitrage_sensitivity(books, thresholds=(1.0, 2.0, 5.0, 30.0)):
    """Re-run the check at several freshness thresholds and summarise.

    If violations only appear at loose thresholds, they are artifacts of the
    feed going quiet rather than real market failures -- which is exactly what
    this table is for.
    """
    rows = []
    for t in thresholds:
        r = check_arbitrage_violations(books, freshness_seconds=t)
        worst = r["violations"]["violation_cents"].max() if r["n_violations"] else 0.0
        rows.append({
            "threshold_s":   t,
            "comparisons":   r["n_comparisons"],
            "violations":    r["n_violations"],
            "violation_%":   100 * r["n_violations"] / max(r["n_comparisons"], 1),
            "worst_cents":   worst,
            "tightest_slack_cents": r["slack"]["tightest_slack_cents"].min(),
        })
    return pd.DataFrame(rows).set_index("threshold_s")


# ============================================================================
# Q2 -- "At what 1-minute interval would you be least certain about how many
# runs there will be in the game?"
#
# These functions are for q2_least_certain_num_of_runs.ipynb ONLY. They are a
# separate set from the Q1 metrics above; do not mix them.
#
# The idea: the KXMLBTOTAL chain is a ladder of survival probabilities
# S(n) = P(runs >= n). Differencing adjacent strikes turns it into a PMF over
# game run totals, from which entropy, E[runs] and variance follow. The PMF
# needs all 11 strikes AT ONCE, so it is only computed when every strike was
# quoted within a freshness threshold -- max gates, mean describes.
# ============================================================================

TOTAL_CHAIN = "TOTAL"          # chain name returned by parse_chain_and_strike
Q2_FRESHNESS_SECONDS = 10.0    # default gate: every strike must be this fresh
Q2_TAIL_VALUE = 13.0           # numeric label for the open-ended ">= 12" bucket


# ---------------------------------------------------------------- layer 1 ---
# Chain extraction: get the strike ladder, and read it out of the walk's cache.

def total_chain_strikes(books):
    """Discover the KXMLBTOTAL strike ladder present in the book feed.

    Selects only the KXMLBTOTAL chain. This matters: KXMLBF5TOTAL,
    KXMLBTEAMTOTAL and KXMLBRFI all begin with 'KXMLB', and a loose substring
    match would silently pull in strikes belonging to a different random
    variable. parse_chain_and_strike names the chain explicitly, so we keep
    only chain == 'TOTAL'.

    Input : books -- DataFrame of order book rows, needs column 'native_id'
    Output: sorted list of int strikes, e.g. [2, 3, ..., 12]

    Asserts the strikes are contiguous. A missing middle strike would make
    differencing silently wrong rather than raise.
    """
    strikes = set()
    for native_id in books["native_id"].unique():
        chain, strike = parse_chain_and_strike(market_short_name(native_id))
        if chain == TOTAL_CHAIN:
            strikes.add(strike)

    strikes = sorted(strikes)
    assert strikes, "no KXMLBTOTAL markets found in the book feed"
    assert strikes == list(range(strikes[0], strikes[-1] + 1)), (
        f"KXMLBTOTAL strikes are not contiguous: {strikes}")
    return strikes


def get_total_ladder(mids, strikes):
    """Read the survival ladder out of the cache, ordered by strike ascending.

    Input : mids    -- dict {short_name: latest mid price}, the walk's cache
            strikes -- the list from total_chain_strikes
    Output: np.ndarray of 11 mid prices, [S(2), S(3), ..., S(12)]

    Asserts every strike is present. A short ladder produces a PMF that looks
    perfectly well formed and is wrong, so this fails loudly rather than
    returning something shorter.
    """
    names = [f"KXMLB{TOTAL_CHAIN}-{k}" for k in strikes]
    missing = [n for n in names if n not in mids]
    assert not missing, f"ladder incomplete, missing {missing}"
    return np.array([mids[n] for n in names], dtype=float)


# ---------------------------------------------------------------- layer 2 ---
# Distribution maths. Pure functions on numpy arrays -- no pandas, no state.

def ladder_to_pmf(S):
    """Difference a survival ladder into a probability mass function.

    S(n) = P(runs >= n), so adjacent differences give P(runs = exactly n),
    with a censored lump at each end that the ladder cannot resolve:

        pmf = [1 - S[0]]  +  [S[i] - S[i+1]]  +  [S[-1]]
            =  P(runs <= 1),   P(runs = 2..11),   P(runs >= 12)

    Input : S   -- np.ndarray of survival probabilities, strike ascending
    Output: np.ndarray of len(S) + 1 bucket probabilities

    The sum telescopes to exactly 1, so no renormalising is needed. Q1 proved
    this chain is monotone everywhere, so no bucket can come out negative.
    Both are asserted anyway -- if either fails, the ladder was assembled in
    the wrong order.
    """
    pmf = np.concatenate([[1.0 - S[0]], -np.diff(S), [S[-1]]])
    assert np.isclose(pmf.sum(), 1.0), f"pmf sums to {pmf.sum()}, not 1"
    assert (pmf >= -1e-12).all(), f"negative pmf bucket: {pmf.min()}"
    return pmf


def pmf_bucket_values(strikes, tail_value=Q2_TAIL_VALUE):
    """The numeric run count each PMF bucket is taken to represent.

    Needed only by the moment calculations. pmf_entropy never calls this,
    which is the entire reason entropy is the primary metric: the open-ended
    P(runs >= 12) bucket holds ~15% of the mass and has no defensible numeric
    value, and entropy is immune to that.

    Input : strikes    -- [2, ..., 12]
            tail_value -- what to call the '>= 12' bucket, default 13.0
    Output: np.ndarray of len(strikes) + 1 values, [1, 2, ..., 11, tail_value]

    The head lump 'runs <= 1' is assigned 1. It holds 3.0-3.5% of the mass and
    the 0-versus-1 split is unknowable, but at that size it cannot move the
    answer.
    """
    head = float(strikes[0] - 1)                      # the "runs <= 1" lump
    interior = [float(k) for k in strikes[:-1]]       # exactly 2 .. exactly 11
    return np.array([head] + interior + [float(tail_value)])


def pmf_entropy(pmf):
    """Shannon entropy of the PMF, in bits: -sum(p * log2(p)).

    Zero-probability buckets contribute nothing and are skipped rather than
    producing -inf. Invariant to what the buckets are CALLED -- it only sees
    how mass is spread across them -- so it needs no tail assignment at all.

    Input : pmf -- np.ndarray of bucket probabilities
    Output: float, entropy in bits
    """
    p = pmf[pmf > 0]
    return float(-(p * np.log2(p)).sum())


def pmf_mean_and_variance(pmf, values):
    """E[runs] and Var[runs] together, since both need the same bucket values.

    Input : pmf    -- np.ndarray of bucket probabilities
            values -- np.ndarray of numeric run counts from pmf_bucket_values
    Output: (mean, variance) as floats

    Var = sum(v^2 * p) - mean^2. Both figures inherit the tail_value
    assumption, which is why the sensitivity run varies it over 12 / 13 / 14.
    """
    mean = float((pmf * values).sum())
    variance = float((pmf * values ** 2).sum() - mean ** 2)
    return mean, variance


# ---------------------------------------------------------------- layer 3 ---
# The walk: one chronological pass over the chain, event by event.

def walk_total_chain(books, freshness_seconds=Q2_FRESHNESS_SECONDS,
                     tail_value=Q2_TAIL_VALUE):
    """One chronological pass over the KXMLBTOTAL chain, event by event.

    At each arriving chain event: update that strike's cache entry with its
    new mid and timestamp, then derive every strike's age as
    (this event's timestamp - its last_seen). Mean age is recorded
    unconditionally -- that is metric (3). The PMF is built only if the MAX age
    is within freshness_seconds: max gates, mean describes.

    Nothing is evaluated until all 11 strikes have been seen at least once
    (warm-up). There is no resampling and no forward fill -- the cache is a
    lookup of what we last observed, and the gate is what decides whether that
    is recent enough to use.

    Input : books             -- full order book DataFrame
            freshness_seconds -- the gate, default 10.0
            tail_value        -- passed to pmf_bucket_values, default 13.0
    Output: DataFrame indexed by recv_ts_utc, one row per chain event, columns
                mean_staleness  float, seconds, always populated after warm-up
                max_staleness   float, seconds, always populated after warm-up
                entropy         float or NaN
                variance        float or NaN
                E_runs          float or NaN
    """
    strikes = total_chain_strikes(books)
    values = pmf_bucket_values(strikes, tail_value)
    names = [f"KXMLB{TOTAL_CHAIN}-{k}" for k in strikes]
    n_strikes = len(names)

    # Slice the chain out once, in time order, with mids precomputed.
    b = books.copy()
    b["market"] = b["native_id"].map(market_short_name)
    b = b[b["market"].isin(names)].copy()
    b["recv_ts_utc"] = pd.to_datetime(b["recv_ts_utc"], utc=True)
    b = b.sort_values("recv_ts_utc")
    b["mid"] = (b["best_bid"] + b["best_ask"]) / 2.0

    last_seen = {}     # market -> np.datetime64 of its most recent quote
    mids = {}          # market -> that quote's mid price
    rows = []

    for ts, market, mid in zip(b["recv_ts_utc"].values,
                               b["market"].values,
                               b["mid"].values):
        last_seen[market] = ts
        mids[market] = mid

        # Warm-up: we cannot speak about the ladder until we have seen all of it
        if len(last_seen) < n_strikes:
            continue

        # Ages are always DERIVED from the stored timestamps, never stored
        ages = np.array([(ts - last_seen[n]) / np.timedelta64(1, "s")
                         for n in names])
        mean_age, max_age = float(ages.mean()), float(ages.max())

        entropy = variance = e_runs = np.nan
        if max_age <= freshness_seconds:                 # <- the gate
            pmf = ladder_to_pmf(get_total_ladder(mids, strikes))
            entropy = pmf_entropy(pmf)
            e_runs, variance = pmf_mean_and_variance(pmf, values)

        rows.append((ts, mean_age, max_age, entropy, variance, e_runs))

    events = pd.DataFrame(rows, columns=["recv_ts_utc", "mean_staleness",
                                         "max_staleness", "entropy",
                                         "variance", "E_runs"])
    # Timestamps came off .values as naive datetime64; they are UTC, so say so.
    # Without this the index prints without a zone and will not compare against
    # tz-aware Timestamps.
    events["recv_ts_utc"] = pd.to_datetime(events["recv_ts_utc"], utc=True)
    return events.set_index("recv_ts_utc")


# ---------------------------------------------------------------- layer 4 ---
# Aggregate the event stream to one row per minute, then report in two tiers.

def minute_frame(events):
    """Reduce the event-level frame to one row per one-minute interval.

    entropy, variance and E_runs take the LAST valid observation inside the
    minute -- the closest analogue to a closing print, and it keeps the
    first-difference series interpretable as minute-to-minute change. A mean
    would blur the very repricing we are trying to locate. mean_staleness is
    the mean over every event in the minute.

    Where no valid PMF could be built anywhere inside the minute, the three
    PMF columns are NaN. Those NaNs are not missing data to be patched; they
    are the primary result -- they mark the intervals where the market could
    not be observed well enough to state an expected number of runs at all.

    Input : events -- the DataFrame from walk_total_chain
    Output: DataFrame indexed by minute, columns
                E_runs, entropy, variance, mean_staleness
    """
    grouped = events.resample("1min")

    # .last() on a resampler skips NaN, so this is the last VALID observation
    minutes = pd.DataFrame({
        "E_runs":         grouped["E_runs"].last(),
        "entropy":        grouped["entropy"].last(),
        "variance":       grouped["variance"].last(),
        "mean_staleness": grouped["mean_staleness"].mean(),
    })
    return minutes


def tier1_minutes(minutes):
    """The minutes where no PMF could be built at all, most uncertain first.

    These are the intervals where the chain was too stale to state an expected
    number of runs -- not imprecisely, but at all -- and they are the
    top-level answer to Q2. Ranked by mean_staleness descending, because max
    cannot rank them: among minutes that already failed the gate it is driven
    by whichever single strike lagged worst, which is the case-A situation the
    plan rejects.

    Input : minutes -- the DataFrame from minute_frame
    Output: DataFrame of the all-NaN minutes, sorted by mean_staleness desc
    """
    tier1 = minutes[minutes["E_runs"].isna()]
    return tier1.sort_values("mean_staleness", ascending=False)


def tier2_minutes(minutes):
    """Among the minutes that do carry a PMF, rank them two ways.

    (a) by entropy descending -- the widest belief over outcomes
    (b) by |change in E_runs from the previous valid minute| descending -- the
        largest repricing. This is a first difference ALONG the series, not a
        dispersion statistic within the bin, because within-minute dispersion
        is exactly zero in 97% of minutes on this data.

    Input : minutes -- the DataFrame from minute_frame
    Output: DataFrame of the valued minutes with an added d_E_runs column
    """
    tier2 = minutes[minutes["E_runs"].notna()].copy()
    # diff() over the valid rows only, so the gap skips the NaN stretches
    tier2["d_E_runs"] = tier2["E_runs"].diff()
    return tier2


def plot_q2_series(minutes, figsize=(13, 9)):
    """Plot the four minute-level columns as four stacked time series.

    Gaps are left as gaps: nothing is interpolated across the NaN stretches,
    because those stretches ARE the Tier 1 answer. Markers are drawn as well
    as lines so that an isolated valid minute surrounded by NaN is still
    visible -- with a line alone it would have no segment to draw and would
    vanish.

    Input : minutes -- the DataFrame from minute_frame
    Output: (fig, axes)
    """
    import matplotlib.pyplot as plt

    panels = [
        ("E_runs",         "E[runs]",                    "tab:blue"),
        ("entropy",        "entropy (bits)",             "tab:orange"),
        ("variance",       "variance",                   "tab:green"),
        ("mean_staleness", "mean staleness (seconds)",   "tab:red"),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=figsize, sharex=True)
    for ax, (col, label, colour) in zip(axes, panels):
        ax.plot(minutes.index, minutes[col], color=colour,
                linewidth=0.9, marker=".", markersize=3)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        n_missing = int(minutes[col].isna().sum())
        ax.set_title(f"{col}   ({len(minutes) - n_missing} of {len(minutes)} "
                     f"minutes populated, {n_missing} blank)",
                     fontsize=9, loc="left")

    axes[-1].set_xlabel("recv_ts_utc")
    fig.suptitle("Q2: uncertainty measures on the KXMLBTOTAL chain, "
                 "one point per minute", y=0.995)
    fig.tight_layout()
    return fig, axes


def q2_threshold_sensitivity(books, thresholds=(1.0, 2.0, 5.0, 10.0, 30.0, 60.0)):
    """Re-run the whole walk at several freshness thresholds and summarise.

    The 10-second default is a judgement call, so the answer's dependence on it
    is reported rather than assumed away -- the same discipline as the Q1
    arbitrage sensitivity table.

    Input : books      -- full order book DataFrame
            thresholds -- seconds to test
    Output: DataFrame, one row per threshold
    """
    rows = []
    for thr in thresholds:
        events = walk_total_chain(books, freshness_seconds=thr)
        minutes = minute_frame(events)
        valid = events["E_runs"].notna()
        rows.append({
            "threshold_s":        thr,
            "valid_instants":     int(valid.sum()),
            "pct_of_events":      100.0 * valid.mean(),
            "minutes_with_value": int(minutes["E_runs"].notna().sum()),
            "tier1_minutes":      int(minutes["E_runs"].isna().sum()),
            "entropy_min":        events["entropy"].min(),
            "entropy_max":        events["entropy"].max(),
            "E_runs_min":         events["E_runs"].min(),
            "E_runs_max":         events["E_runs"].max(),
        })
    return pd.DataFrame(rows).set_index("threshold_s")
