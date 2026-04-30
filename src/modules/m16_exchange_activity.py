"""M16: Cross-Exchange Activity — Funding spread, OI divergence, L/S divergence.

Pulls data from Binance, OKX, and Bybit to detect:
- Funding rate arbitrage / divergence across exchanges
- OI migration between exchanges (capital flow)
- L/S ratio divergence (different positioning per exchange)
- Aggregate exchange flow bias
"""

import numpy as np
import pandas as pd
import ccxt
from concurrent.futures import ThreadPoolExecutor, as_completed

SYMBOL = "ETH/USDT:USDT"
EXCHANGES = ["binance", "okx", "bybit"]
EXCHANGES_EXTENDED = ["binance", "okx", "bybit", "htx", "phemex", "kraken"]
FUNDING_HISTORY_LIMIT = 30  # ~3.75 days of 8h funding

# Kraken uses ETH/USD:USD instead of ETH/USDT:USDT
SYMBOL_MAP = {
    "kraken": "ETH/USD:USD",
}


def _make_exchange(name):
    """Create ccxt exchange instance."""
    # Some exchanges have different class names
    CLASS_MAP = {
        "kraken": "krakenfutures",
        "kucoin": "kucoinfutures",
    }
    cls_name = CLASS_MAP.get(name, name)
    cls = getattr(ccxt, cls_name)
    return cls({"options": {"defaultType": "swap"}, "enableRateLimit": True})


def _fetch_snapshot(name):
    """Fetch current funding, OI, and L/S ratio from one exchange."""
    ex = _make_exchange(name)
    symbol = SYMBOL_MAP.get(name, SYMBOL)
    result = {"exchange": name}

    try:
        fr = ex.fetch_funding_rate(symbol)
        result["funding_rate"] = float(fr.get("fundingRate", 0))
        result["mark_price"] = float(fr.get("markPrice", 0))
    except Exception as e:
        result["funding_rate"] = None
        result["funding_error"] = str(e)

    try:
        oi = ex.fetch_open_interest(symbol)
        oi_raw = float(oi.get("openInterestAmount", 0))
        # Some exchanges return OI in contracts, need to convert to ETH
        # Known contract sizes
        CONTRACT_SIZES = {
            "htx": 0.01,      # 1 contract = 0.01 ETH
            "kucoin": 0.01,   # 1 contract = 0.01 ETH
            "bitmex": 0.0001, # 1 contract = 0.0001 ETH (satoshi-denominated)
        }
        contract_size = CONTRACT_SIZES.get(name, 1.0)
        result["oi"] = oi_raw * contract_size
        result["oi_raw"] = oi_raw
        result["oi_unit"] = "contracts" if contract_size != 1.0 else "ETH"
        oi_usd = oi.get("openInterestValue")
        result["oi_usd"] = float(oi_usd) if oi_usd else None
    except Exception as e:
        result["oi"] = None
        result["oi_error"] = str(e)

    try:
        if hasattr(ex, "fetch_long_short_ratio_history"):
            ls = ex.fetch_long_short_ratio_history(symbol, limit=1)
            if ls:
                ls_val = float(ls[-1].get("longShortRatio", 0))
                # Bitget L/S is broken (returns ~0.024), skip unreliable values
                if ls_val < 0.1:
                    result["ls_ratio"] = None
                    result["ls_error"] = "unreliable_value"
                else:
                    result["ls_ratio"] = ls_val
            else:
                result["ls_ratio"] = None
        else:
            result["ls_ratio"] = None
    except Exception as e:
        result["ls_ratio"] = None
        result["ls_error"] = str(e)

    return result


def _fetch_funding_history(name, limit=FUNDING_HISTORY_LIMIT):
    """Fetch historical funding rates from one exchange."""
    ex = _make_exchange(name)
    symbol = SYMBOL_MAP.get(name, SYMBOL)
    try:
        if hasattr(ex, "fetch_funding_rate_history"):
            hist = ex.fetch_funding_rate_history(symbol, limit=limit)
            rows = []
            for h in hist:
                rows.append({
                    "timestamp": pd.to_datetime(h["timestamp"], unit="ms"),
                    "exchange": name,
                    "funding_rate": float(h["fundingRate"]),
                })
            return pd.DataFrame(rows)
    except Exception:
        pass
    return pd.DataFrame()


def _fetch_oi_history(name, timeframe="1h", limit=48):
    """Fetch OI history from one exchange (Binance/Bybit have this)."""
    ex = _make_exchange(name)
    symbol = SYMBOL_MAP.get(name, SYMBOL)
    CONTRACT_SIZES = {"htx": 0.01, "kucoin": 0.01, "bitmex": 0.0001}
    contract_size = CONTRACT_SIZES.get(name, 1.0)
    try:
        if hasattr(ex, "fetch_open_interest_history"):
            hist = ex.fetch_open_interest_history(symbol, timeframe=timeframe, limit=limit)
            rows = []
            for h in hist:
                oi_val = h.get("openInterestAmount")
                if oi_val is not None:
                    rows.append({
                        "timestamp": pd.to_datetime(h["timestamp"], unit="ms"),
                        "exchange": name,
                        "oi": float(oi_val) * contract_size,
                    })
            return pd.DataFrame(rows)
    except Exception:
        pass
    return pd.DataFrame()


def fetch_all_exchange_data():
    """Fetch all exchange data in parallel for speed."""
    snapshots = {}
    funding_dfs = []
    oi_dfs = []

    # Parallel snapshots (all exchanges)
    with ThreadPoolExecutor(max_workers=len(EXCHANGES_EXTENDED)) as pool:
        futures = {pool.submit(_fetch_snapshot, name): name for name in EXCHANGES_EXTENDED}
        for f in as_completed(futures):
            name = futures[f]
            try:
                snapshots[name] = f.result()
            except Exception:
                snapshots[name] = {"exchange": name, "error": "fetch failed"}

    # Historical funding (parallel — all exchanges)
    with ThreadPoolExecutor(max_workers=len(EXCHANGES_EXTENDED)) as pool:
        futures = {pool.submit(_fetch_funding_history, name): name for name in EXCHANGES_EXTENDED}
        for f in as_completed(futures):
            try:
                df = f.result()
                if not df.empty:
                    funding_dfs.append(df)
            except Exception:
                pass

    # OI history (parallel — Binance + Bybit only, OKX/HTX/Phemex may return None)
    oi_exchanges = ["binance", "bybit"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_fetch_oi_history, name): name for name in oi_exchanges}
        for f in as_completed(futures):
            try:
                df = f.result()
                if not df.empty:
                    oi_dfs.append(df)
            except Exception:
                pass

    funding_df = pd.concat(funding_dfs, ignore_index=True) if funding_dfs else pd.DataFrame()
    oi_df = pd.concat(oi_dfs, ignore_index=True) if oi_dfs else pd.DataFrame()

    return snapshots, funding_df, oi_df


def _fetch_spot_data(name):
    """Fetch spot market data: ticker, order book, recent trades."""
    CLASS_MAP = {"kraken": "krakenfutures", "kucoin": "kucoinfutures"}
    cls_name = CLASS_MAP.get(name, name)
    ex = getattr(ccxt, cls_name)({"enableRateLimit": True})
    result = {"exchange": name}

    # Spot symbol — most use ETH/USDT, some use ETH/USD
    SPOT_SYMBOLS = {"kraken": "ETH/USD", "coinbase": "ETH/USD"}
    spot_sym = SPOT_SYMBOLS.get(name, "ETH/USDT")

    # Ticker
    try:
        t = ex.fetch_ticker(spot_sym)
        result["spot_price"] = float(t.get("last", 0))
        result["spot_vol_24h"] = float(t.get("baseVolume", 0))
        result["spot_change_24h"] = float(t.get("percentage", 0)) if t.get("percentage") else None
        result["spot_high"] = float(t.get("high", 0))
        result["spot_low"] = float(t.get("low", 0))
    except Exception as e:
        result["spot_error"] = str(e)

    # Order book
    try:
        ob = ex.fetch_order_book(spot_sym, limit=20)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        bid_vol = sum(b[1] for b in bids[:20]) if bids else 0
        ask_vol = sum(a[1] for a in asks[:20]) if asks else 0
        result["spot_bid_vol"] = round(bid_vol, 2)
        result["spot_ask_vol"] = round(ask_vol, 2)
        result["spot_ob_ratio"] = round(bid_vol / ask_vol, 4) if ask_vol > 0 else 0

        # Find largest single wall
        if bids:
            max_bid = max(bids, key=lambda x: x[1])
            result["spot_max_bid"] = {"price": float(max_bid[0]), "size": round(float(max_bid[1]), 2)}
        if asks:
            max_ask = max(asks, key=lambda x: x[1])
            result["spot_max_ask"] = {"price": float(max_ask[0]), "size": round(float(max_ask[1]), 2)}
    except Exception:
        pass

    # Recent trades (last 100)
    try:
        trades = ex.fetch_trades(spot_sym, limit=100)
        buy_vol = sum(float(t["amount"]) for t in trades if t.get("side") == "buy")
        sell_vol = sum(float(t["amount"]) for t in trades if t.get("side") == "sell")
        total = buy_vol + sell_vol
        result["spot_trade_buy"] = round(buy_vol, 2)
        result["spot_trade_sell"] = round(sell_vol, 2)
        result["spot_trade_ratio"] = round(buy_vol / sell_vol, 4) if sell_vol > 0 else 0
        result["spot_buy_pct"] = round(buy_vol / total * 100, 1) if total > 0 else 50
    except Exception:
        pass

    return result


def fetch_all_spot_data():
    """Fetch spot data from all exchanges in parallel."""
    spot_exchanges = ["binance", "okx", "bybit", "kraken", "coinbase", "htx"]
    results = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch_spot_data, name): name for name in spot_exchanges}
        for f in as_completed(futures):
            name = futures[f]
            try:
                results[name] = f.result()
            except Exception:
                results[name] = {"exchange": name, "error": "fetch failed"}
    return results


def compute_spot_signals(spot_data, snapshots):
    """Compute spot-based signals."""
    signals = {}

    # --- Spot-Futures Basis ---
    basis = {}
    for name, s in spot_data.items():
        spot_price = s.get("spot_price")
        fut_price = snapshots.get(name, {}).get("mark_price")
        if spot_price and fut_price and spot_price > 0:
            b = (fut_price - spot_price) / spot_price * 100
            basis[name] = round(b, 4)
    signals["basis"] = basis

    if basis:
        avg_basis = np.mean(list(basis.values()))
        signals["basis_avg"] = round(avg_basis, 4)
        signals["basis_state"] = "BACKWARDATION" if avg_basis < -0.02 else (
            "CONTANGO" if avg_basis > 0.02 else "FLAT")

        # Is any exchange an outlier?
        std = np.std(list(basis.values())) if len(basis) > 1 else 0
        for ex, b in basis.items():
            if std > 0 and abs(b - avg_basis) > 2 * std:
                signals[f"basis_outlier_{ex}"] = b

    # --- Spot Volume Share ---
    volumes = {name: s.get("spot_vol_24h", 0) for name, s in spot_data.items()
               if s.get("spot_vol_24h", 0) > 0}
    if volumes:
        total_vol = sum(volumes.values())
        vol_shares = {name: vol / total_vol for name, vol in volumes.items()}
        signals["spot_vol_shares"] = {k: round(v, 4) for k, v in vol_shares.items()}
        signals["spot_vol_total"] = round(total_vol, 0)
        signals["spot_vol_dominant"] = max(vol_shares, key=vol_shares.get)

    # --- Spot Order Book Imbalance ---
    ob_ratios = {name: s.get("spot_ob_ratio", 0) for name, s in spot_data.items()
                 if s.get("spot_ob_ratio", 0) > 0}
    if ob_ratios:
        avg_ratio = np.mean(list(ob_ratios.values()))
        signals["spot_ob_ratios"] = {k: round(v, 4) for k, v in ob_ratios.items()}
        signals["spot_ob_avg"] = round(avg_ratio, 4)

        # Heavy sell wall detection (ratio < 0.3)
        sell_walls = {ex: r for ex, r in ob_ratios.items() if r < 0.3}
        signals["spot_sell_walls"] = list(sell_walls.keys())

        # Strong bid support (ratio > 1.5)
        bid_support = {ex: r for ex, r in ob_ratios.items() if r > 1.5}
        signals["spot_bid_support"] = list(bid_support.keys())

        # Largest single walls
        for ex, s in spot_data.items():
            max_ask = s.get("spot_max_ask", {})
            if max_ask and max_ask.get("size", 0) > 50:
                signals[f"spot_wall_{ex}"] = {
                    "price": max_ask["price"],
                    "size": max_ask["size"],
                    "side": "ASK"
                }

    # --- Spot Trade Flow ---
    trade_ratios = {name: s.get("spot_trade_ratio", 0) for name, s in spot_data.items()
                    if s.get("spot_trade_ratio", 0) > 0}
    if trade_ratios:
        avg_tr = np.mean(list(trade_ratios.values()))
        signals["spot_trade_ratios"] = {k: round(v, 4) for k, v in trade_ratios.items()}
        signals["spot_trade_avg"] = round(avg_tr, 4)
        signals["spot_flow"] = "BUYERS" if avg_tr > 1.1 else (
            "SELLERS" if avg_tr < 0.9 else "NEUTRAL")

        # Per-exchange flow
        for ex, r in trade_ratios.items():
            if r > 1.3:
                signals[f"spot_flow_{ex}"] = "BUYERS"
            elif r < 0.7:
                signals[f"spot_flow_{ex}"] = "SELLERS"

    # --- Spot vs Futures Volume Ratio ---
    for name, s in spot_data.items():
        spot_vol = s.get("spot_vol_24h", 0)
        fut_vol = snapshots.get(name, {}).get("oi", 0)  # approximate
        # Use a rough estimate: if we have spot vol, compare
        if spot_vol and spot_vol > 0:
            signals[f"spot_vol_{name}"] = round(spot_vol, 0)

    return signals


def score_spot_signals(signals, direction):
    """Score spot signals for a given trade direction."""
    score = 0.5
    details = {}
    factors = []

    # 1. Spot-Futures Basis
    basis_avg = signals.get("basis_avg", 0)
    basis_state = signals.get("basis_state", "FLAT")
    if direction == "LONG":
        if basis_state == "BACKWARDATION":
            score -= 0.05
            factors.append(f"futures discount {basis_avg:.4f}% (bearish sentiment)")
        elif basis_state == "CONTANGO":
            score += 0.03
            factors.append(f"futures premium {basis_avg:.4f}%")
    elif direction == "SHORT":
        if basis_state == "BACKWARDATION":
            score += 0.03
            factors.append(f"futures discount {basis_avg:.4f}% (confirms bearish)")
        elif basis_state == "CONTANGO":
            score -= 0.03
            factors.append(f"futures premium {basis_avg:.4f}%")
    details["basis_avg"] = basis_avg
    details["basis_state"] = basis_state

    # 2. Spot Order Book Imbalance
    ob_avg = signals.get("spot_ob_avg", 1.0)
    sell_walls = signals.get("spot_sell_walls", [])
    bid_support = signals.get("spot_bid_support", [])

    if direction == "LONG":
        if ob_avg < 0.3:
            score -= 0.08
            factors.append(f"heavy spot sell wall (ratio={ob_avg:.3f})")
        elif ob_avg > 1.0:
            score += 0.05
            factors.append(f"spot bids dominate (ratio={ob_avg:.3f})")
        if sell_walls:
            factors.append(f"sell walls on: {', '.join(sell_walls)}")
    elif direction == "SHORT":
        if ob_avg > 1.5:
            score -= 0.05
            factors.append(f"strong spot bid support (ratio={ob_avg:.3f})")
        elif ob_avg < 0.5:
            score += 0.05
            factors.append(f"spot asks dominate (ratio={ob_avg:.3f})")
    details["spot_ob_avg"] = ob_avg
    details["spot_sell_walls"] = sell_walls

    # 3. Spot Trade Flow
    flow = signals.get("spot_flow", "NEUTRAL")
    trade_avg = signals.get("spot_trade_avg", 1.0)
    if direction == "LONG":
        if flow == "BUYERS":
            score += 0.05
            factors.append(f"spot flow: buyers ({trade_avg:.2f})")
        elif flow == "SELLERS":
            score -= 0.03
            factors.append(f"spot flow: sellers ({trade_avg:.2f})")
    elif direction == "SHORT":
        if flow == "SELLERS":
            score += 0.05
            factors.append(f"spot flow: sellers ({trade_avg:.2f})")
        elif flow == "BUYERS":
            score -= 0.03
            factors.append(f"spot flow: buyers ({trade_avg:.2f})")
    details["spot_flow"] = flow
    details["spot_trade_avg"] = trade_avg

    # 4. Spot walls — large ask walls are resistance
    for key, val in signals.items():
        if key.startswith("spot_wall_") and isinstance(val, dict):
            if val.get("side") == "ASK" and val.get("size", 0) > 100:
                factors.append(f"{key.replace('spot_wall_', '')} wall: {val['size']:.0f} ETH @ ${val['price']:.0f}")

    score = max(0.0, min(1.0, score))
    status = "PASS" if score >= 0.40 else "FAIL"
    details["spot_score"] = round(score, 3)
    details["factors"] = factors

    return status, score, details


def compute_exchange_signals(snapshots, funding_df, oi_df):
    """Compute cross-exchange divergence signals."""
    signals = {}

    # --- Current snapshot signals ---
    rates = {name: s.get("funding_rate") for name, s in snapshots.items()
             if s.get("funding_rate") is not None}
    ois = {name: s.get("oi") for name, s in snapshots.items()
           if s.get("oi") is not None and s["oi"] > 0}
    ls_ratios = {name: s.get("ls_ratio") for name, s in snapshots.items()
                 if s.get("ls_ratio") is not None}

    # Funding spread
    if len(rates) >= 2:
        values = list(rates.values())
        signals["funding_spread"] = round(max(values) - min(values), 8)
        signals["funding_spread_exchanges"] = (
            max(rates, key=rates.get), min(rates, key=rates.get))
        signals["funding_by_exchange"] = {k: round(v, 8) for k, v in rates.items()}

        # Mean funding across exchanges
        signals["funding_mean"] = round(np.mean(values), 8)

        # Is any exchange an outlier? (>2x the spread from mean)
        mean = np.mean(values)
        std = np.std(values) if len(values) > 1 else 0
        outliers = {}
        for ex, rate in rates.items():
            if std > 0 and abs(rate - mean) > 2 * std:
                outliers[ex] = round(rate, 8)
        signals["funding_outliers"] = outliers

    # OI divergence (capital migration)
    if len(ois) >= 2:
        # Normalize to total OI share
        total = sum(ois.values())
        oi_shares = {name: oi / total for name, oi in ois.items()}
        signals["oi_shares"] = {k: round(v, 4) for k, v in oi_shares.items()}
        signals["oi_total"] = round(total, 0)

        # Dominant exchange
        dominant = max(oi_shares, key=oi_shares.get)
        signals["oi_dominant_exchange"] = dominant

        # OI concentration (HHI-like)
        hhi = sum(s ** 2 for s in oi_shares.values())
        signals["oi_concentration"] = round(hhi, 4)  # 1.0 = all on one, 0.33 = equal

    # L/S ratio divergence
    if len(ls_ratios) >= 2:
        values = list(ls_ratios.values())
        signals["ls_spread"] = round(max(values) - min(values), 4)
        signals["ls_by_exchange"] = {k: round(v, 4) for k, v in ls_ratios.items()}

        # Most bullish / most bearish exchange
        signals["ls_most_bullish"] = max(ls_ratios, key=ls_ratios.get)
        signals["ls_most_bearish"] = min(ls_ratios, key=ls_ratios.get)

    # --- Historical signals ---
    # Funding trend: is funding converging or diverging across exchanges?
    if not funding_df.empty and len(funding_df) >= 6:
        pivot = funding_df.pivot_table(
            index="timestamp", columns="exchange", values="funding_rate")
        pivot = pivot.ffill().dropna(axis=1, how="all")

        if len(pivot.columns) >= 2:
            # Rolling spread
            spread_series = pivot.max(axis=1) - pivot.min(axis=1)
            if len(spread_series) >= 8:
                recent_spread = spread_series.iloc[-4:].mean()
                older_spread = spread_series.iloc[-12:-4].mean() if len(spread_series) >= 12 else recent_spread
                signals["funding_spread_trend"] = "DIVERGING" if recent_spread > older_spread * 1.3 else (
                    "CONVERGING" if recent_spread < older_spread * 0.7 else "STABLE")
                signals["funding_spread_recent"] = round(recent_spread, 8)
                signals["funding_spread_older"] = round(older_spread, 8)

            # Per-exchange funding trend (bullish/bearish shift)
            for col in pivot.columns:
                series = pivot[col].dropna()
                if len(series) >= 8:
                    recent = series.iloc[-4:].mean()
                    older = series.iloc[-8:-4].mean()
                    trend = "RISING" if recent > older * 1.5 and recent > 0 else (
                        "FALLING" if recent < older * 0.5 and recent < 0 else "STABLE")
                    signals[f"funding_trend_{col}"] = trend

    # OI trend across exchanges
    if not oi_df.empty and len(oi_df) >= 6:
        for ex in oi_df["exchange"].unique():
            ex_data = oi_df[oi_df["exchange"] == ex].sort_values("timestamp")
            if len(ex_data) >= 8:
                recent_oi = ex_data["oi"].iloc[-4:].mean()
                older_oi = ex_data["oi"].iloc[-8:-4].mean()
                roc = (recent_oi - older_oi) / older_oi * 100 if older_oi > 0 else 0
                signals[f"oi_roc_{ex}"] = round(roc, 3)

        # Cross-exchange OI migration
        rocs = {k: v for k, v in signals.items() if k.startswith("oi_roc_")}
        if len(rocs) >= 2:
            roc_vals = list(rocs.values())
            migration_spread = max(roc_vals) - min(roc_vals)
            signals["oi_migration_spread"] = round(migration_spread, 3)
            if migration_spread > 3.0:
                gainer = max(rocs, key=rocs.get).replace("oi_roc_", "")
                loser = min(rocs, key=rocs.get).replace("oi_roc_", "")
                signals["oi_migration"] = f"{gainer}↑ {loser}↓"
            else:
                signals["oi_migration"] = "BALANCED"

    return signals


def score_exchange_activity(signals, direction):
    """Score exchange activity signals for a given trade direction.

    Returns (status, score, details).
    """
    score = 0.5
    details = {}
    factors = []

    # 1. Funding spread — wide spread = arbitrage opportunity, can mean instability
    spread = signals.get("funding_spread", 0)
    if spread > 0.0003:  # >3bps spread
        score -= 0.05
        factors.append(f"wide funding spread {spread*100:.3f}%")
    details["funding_spread"] = spread

    # 2. Funding mean — aggregate direction
    funding_mean = signals.get("funding_mean", 0)
    if direction == "LONG":
        if funding_mean < -0.0001:  # Negative funding favors longs
            score += 0.08
            factors.append(f"negative aggregate funding {funding_mean*100:.4f}%")
        elif funding_mean > 0.0005:
            score -= 0.05
            factors.append(f"high positive funding {funding_mean*100:.4f}%")
    elif direction == "SHORT":
        if funding_mean > 0.0001:
            score += 0.08
            factors.append(f"positive aggregate funding {funding_mean*100:.4f}%")
        elif funding_mean < -0.0005:
            score -= 0.05
            factors.append(f"deep negative funding {funding_mean*100:.4f}%")
    details["funding_mean"] = funding_mean

    # 3. Funding trend per exchange
    for ex in EXCHANGES_EXTENDED:
        trend = signals.get(f"funding_trend_{ex}")
        if trend:
            if direction == "LONG" and trend == "FALLING":
                score += 0.02
                factors.append(f"{ex} funding falling (bullish)")
            elif direction == "SHORT" and trend == "RISING":
                score += 0.02
                factors.append(f"{ex} funding rising (bearish)")
    details["funding_trends"] = {ex: signals.get(f"funding_trend_{ex}", "?") for ex in EXCHANGES}

    # 4. OI migration — capital flowing to one exchange
    migration = signals.get("oi_migration", "BALANCED")
    if migration != "BALANCED":
        score += 0.05
        factors.append(f"OI migration: {migration}")
    details["oi_migration"] = migration

    # 5. L/S ratio divergence — different positioning across exchanges
    ls_spread = signals.get("ls_spread", 0)
    if ls_spread > 0.5:
        # Exchanges disagree on positioning — uncertainty
        score -= 0.03
        factors.append(f"L/S divergence {ls_spread:.2f}")
    details["ls_spread"] = ls_spread

    # 6. L/S most bullish/bearish exchange alignment
    ls_bullish = signals.get("ls_most_bullish")
    ls_bearish = signals.get("ls_most_bearish")
    if direction == "LONG" and ls_bullish:
        # Check if the exchange with highest L/S is the one we're trading on
        score += 0.03
        factors.append(f"most bullish exchange: {ls_bullish}")
    elif direction == "SHORT" and ls_bearish:
        score += 0.03
        factors.append(f"most bearish exchange: {ls_bearish}")
    details["ls_by_exchange"] = signals.get("ls_by_exchange", {})

    # 7. Funding outliers — one exchange pricing differently
    outliers = signals.get("funding_outliers", {})
    if outliers:
        score -= 0.03
        factors.append(f"funding outliers: {outliers}")
    details["funding_outliers"] = outliers

    # 8. OI concentration
    concentration = signals.get("oi_concentration", 0)
    if concentration > 0.6:
        # High concentration = one exchange dominates
        details["oi_concentration"] = concentration
        details["oi_dominant"] = signals.get("oi_dominant_exchange")

    score = max(0.0, min(1.0, score))
    status = "PASS" if score >= 0.45 else "FAIL"
    details["exchange_score"] = round(score, 3)
    details["factors"] = factors

    return status, score, details


def get_exchange_summary():
    """Fetch and summarize cross-exchange data for scan output."""
    try:
        snapshots, funding_df, oi_df = fetch_all_exchange_data()
        signals = compute_exchange_signals(snapshots, funding_df, oi_df)

        # Fetch spot data
        spot_data = fetch_all_spot_data()
        spot_signals = compute_spot_signals(spot_data, snapshots)

        summary = {
            "snapshots": {},
            "signals": signals,
            "spot": {},
            "spot_signals": spot_signals,
        }

        for name in EXCHANGES_EXTENDED:
            snap = snapshots.get(name, {})
            summary["snapshots"][name] = {
                "funding_rate": round(snap.get("funding_rate", 0), 8) if snap.get("funding_rate") is not None else None,
                "oi": round(snap.get("oi", 0), 0) if snap.get("oi") else None,
                "ls_ratio": round(snap.get("ls_ratio", 0), 4) if snap.get("ls_ratio") else None,
                "mark_price": snap.get("mark_price"),
            }

        # Spot data
        for name, s in spot_data.items():
            summary["spot"][name] = {
                "price": s.get("spot_price"),
                "vol_24h": round(s.get("spot_vol_24h", 0), 0) if s.get("spot_vol_24h") else None,
                "ob_ratio": s.get("spot_ob_ratio"),
                "trade_ratio": s.get("spot_trade_ratio"),
                "buy_pct": s.get("spot_buy_pct"),
                "max_ask": s.get("spot_max_ask"),
            }

        return summary
    except Exception as e:
        return {"error": str(e)}
