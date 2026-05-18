"""
M-Signal: Technical Composite — replaces M22 in ICS

Three empirically-validated dimensions (backtested on 286 releases, 2018-2026):
  1. range_pct:   Wider 48-bar range → bullish continuation (r=+0.114, p=0.055)
  2. dist_from_high: Near 48-bar high → bearish reversion (r=-0.104, p=0.080)
  3. macd_hist:   Positive MACD histogram → bullish (r=+0.099, p=0.094)

Plus NFP surprise gate (Spearman r=+0.127, p=0.031):
  |NFP surprise| > 50K from consensus → directional bias

Usage:
    from src.modules.m_signal import score_signal, format_signal
    status, score, details = score_signal(df_15m, idx, direction, config=cfg)
"""

from src.config import CONFIG
import numpy as np


def score_signal(df_15m, idx, direction, nfp_surprise_k=None, config=None):
    """Score the technical composite for ICS.

    Args:
        df_15m: DataFrame with 15m OHLCV data
        idx: Current bar index
        direction: 'LONG' or 'SHORT'
        nfp_surprise_k: NFP surprise in thousands (actual - consensus), or None
        config: Config dict

    Returns:
        status: 'PASS' or 'SKIP'
        score: 0.0-1.0
        details: dict with breakdown
    """
    cfg = config or CONFIG

    if not cfg.get('SIGNAL_ENABLED', True):
        return 'SKIP', 0.5, {'regime': 'DISABLED'}

    if idx < 60:
        return 'SKIP', 0.5, {'regime': 'INSUFFICIENT_DATA'}

    close = df_15m['Close'].values.astype(float)
    high = df_15m['High'].values.astype(float)
    low = df_15m['Low'].values.astype(float)
    price = close[idx]

    # ── Dimension 1: Range % (48-bar) ──
    # Wider range = trend continuation. Backtested: r=+0.114, p=0.055
    roll_high = np.max(high[max(0, idx-47):idx+1])
    roll_low = np.min(low[max(0, idx-47):idx+1])
    range_pct = (roll_high - roll_low) / price * 100

    # Normalize: 1.5% = 0.0 (neutral), 5% = 1.0 (strong bullish), 0.5% = 0.0 (compressed)
    range_score = max(0.0, min(1.0, (range_pct - 0.5) / 4.5))

    # ── Dimension 2: Distance from high (48-bar) ──
    # Near high = bearish reversion. Backtested: r=-0.104, p=0.080
    dist_from_high = (price - roll_high) / roll_high * 100  # negative when below high

    # Normalize: 0% (at high) = 0.0 (bearish), -5% (below high) = 1.0 (bullish)
    # Inverted: being FAR from high is bullish (mean reversion up)
    dist_score = max(0.0, min(1.0, (-dist_from_high - 0.5) / 4.5))

    # ── Dimension 3: MACD histogram ──
    # Positive histogram = bullish. Backtested: r=+0.099, p=0.094
    from src.utils.indicators import calc_macd
    lookback = min(idx + 1, 600)
    c = df_15m['Close'].iloc[max(0, idx-lookback+1):idx+1].astype(float)
    macd_line, macd_signal, macd_hist = calc_macd(
        c, cfg.get('MACD_FAST', 12), cfg.get('MACD_SLOW', 26), cfg.get('MACD_SIGNAL', 9))

    hist_val = float(macd_hist.iloc[-1]) if not np.isnan(macd_hist.iloc[-1]) else 0
    # Normalize: positive = bullish, scale by ATR
    atr_val = float(df_15m['atr'].iloc[idx]) if 'atr' in df_15m.columns and not np.isnan(df_15m['atr'].iloc[idx]) else price * 0.02
    hist_normalized = hist_val / atr_val if atr_val > 0 else 0
    # Clamp to [-2, +2] range
    hist_score = max(0.0, min(1.0, (hist_normalized + 2) / 4))

    # ── Weighted composite ──
    # Weights based on individual predictive power (|r| values)
    # range_pct: 0.114, dist_from_high: 0.104, macd_hist: 0.099
    # Normalized: range=0.363, dist=0.331, macd=0.315 → roughly equal
    w_range = cfg.get('SIGNAL_W_RANGE', 0.36)
    w_dist = cfg.get('SIGNAL_W_DIST', 0.33)
    w_macd = cfg.get('SIGNAL_W_MACD', 0.31)

    raw_score = range_score * w_range + dist_score * w_dist + hist_score * w_macd

    # ── NFP surprise gate ──
    # Spearman r=+0.127, p=0.031. Non-linear: big surprises matter, small ones don't.
    nfp_adj = 0.0
    nfp_active = False
    if nfp_surprise_k is not None and abs(nfp_surprise_k) > 50:
        # Strong NFP: positive surprise = strong labor = Fed can hold = slight bearish for risk
        # Negative surprise = weak labor = Fed must cut = slight bullish for risk
        # But backtest shows big NFP surprises predict positive returns (Spearman +0.127)
        # So: big surprise (either direction) → slight bullish bias
        nfp_adj = 0.05
        nfp_active = True

    raw_score = max(0.0, min(1.0, raw_score + nfp_adj))

    # ── Direction adjustment ──
    # The composite is inherently directional (bullish when high).
    # For SHORT trades, invert.
    if direction == 'SHORT':
        score = 1.0 - raw_score
    else:
        score = raw_score

    score = max(0.0, min(1.0, score))

    # ── Status ──
    if score >= 0.55:
        status = 'PASS'
    else:
        status = 'FAIL'

    details = {
        'range_pct': round(range_pct, 3),
        'range_score': round(range_score, 3),
        'dist_from_high': round(dist_from_high, 3),
        'dist_score': round(dist_score, 3),
        'macd_hist': round(hist_val, 4),
        'macd_hist_normalized': round(hist_normalized, 4),
        'macd_score': round(hist_score, 3),
        'raw_score': round(raw_score, 3),
        'nfp_surprise_k': nfp_surprise_k,
        'nfp_active': nfp_active,
        'nfp_adj': nfp_adj,
        'direction': direction,
        'score': round(score, 3),
        'weights': {'range': w_range, 'dist': w_dist, 'macd': w_macd},
    }

    return status, score, details


def format_signal(details):
    """Format signal details for terminal output."""
    if not details or details.get('regime') in ('DISABLED', 'INSUFFICIENT_DATA'):
        return ''

    lines = []
    lines.append(f"\n  📊 M-SIGNAL (Technical Composite)")

    # Dimensions
    range_pct = details.get('range_pct', 0)
    range_sc = details.get('range_score', 0)
    dist = details.get('dist_from_high', 0)
    dist_sc = details.get('dist_score', 0)
    hist = details.get('macd_hist', 0)
    hist_sc = details.get('macd_score', 0)

    range_icon = '🟢' if range_sc > 0.6 else '🔴' if range_sc < 0.3 else '⚪'
    dist_icon = '🟢' if dist_sc > 0.6 else '🔴' if dist_sc < 0.3 else '⚪'
    hist_icon = '🟢' if hist_sc > 0.6 else '🔴' if hist_sc < 0.3 else '⚪'

    lines.append(f"    {range_icon} Range:     {range_pct:.1f}%  score={range_sc:.3f}")
    lines.append(f"    {dist_icon} DistHigh:  {dist:+.1f}%  score={dist_sc:.3f}")
    lines.append(f"    {hist_icon} MACD hist: {hist:+.2f}  score={hist_sc:.3f}")

    # NFP
    if details.get('nfp_active'):
        nfp_k = details.get('nfp_surprise_k', 0)
        lines.append(f"    💼 NFP surprise gate: {nfp_k:+.0f}K  adj=+0.05")

    # Composite
    raw = details.get('raw_score', 0)
    score = details.get('score', 0)
    direction = details.get('direction', '?')
    lines.append(f"    → Raw: {raw:.3f}  ({direction}) Score: {score:.3f}")

    return '\n'.join(lines)
