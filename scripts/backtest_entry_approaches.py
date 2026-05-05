#!/usr/bin/env python3
"""
Backtest: Squeeze Entry Approaches

Tests different entry methods for squeeze breakouts:
  1. BASELINE    — current: close beyond coil + vol ≥ 1x MA20
  2. RETEST      — breakout → retest coil edge → enter on reclaim
  3. WICK_CONF   — require wick rejection (body > 60% of range) on breakout bar
  4. MOMENTUM    — RSI > 50 for long / < 50 for short on breakout bar
  5. TWO_BAR     — 2 consecutive closes beyond coil
  6. VOL_SPIKE   — vol ≥ 1.5x MA20 on breakout bar
  7. COMBO       — wick_conf + momentum + vol ≥ 1.2x

Usage:
    python scripts/backtest_entry_approaches.py
    python scripts/backtest_entry_approaches.py --year 2026
    python scripts/backtest_entry_approaches.py --lookforward 48
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime

from src.modules.m18_squeeze import (
    _check_compression, _compute_2h_macd, _detect_macd_coil,
    _check_entry_trigger, SQUEEZE_V5_DEFAULTS,
)


# ═══════════════════════════════════════════════════════════════
# ENTRY APPROACHES
# ═══════════════════════════════════════════════════════════════

def entry_baseline(close, high, low, vol, vol_ma20, coil_high, coil_low, direction, cfg):
    """Current approach: close beyond coil + vol ≥ 1x MA20."""
    buf = cfg['SQUEEZE_ENTRY_BUFFER_PCT']
    if direction == 'LONG':
        return close > coil_high * (1 + buf) and vol >= vol_ma20 * 1.0
    else:
        return close < coil_low * (1 - buf) and vol >= vol_ma20 * 1.0


def entry_retest(close, high, low, vol, vol_ma20, coil_high, coil_low, direction, prev_close, prev_high, prev_low, cfg):
    """Breakout → retest → reclaim. Requires previous bar to break out,
    current bar retests the coil edge and closes back outside."""
    buf = cfg['SQUEEZE_ENTRY_BUFFER_PCT']
    if direction == 'LONG':
        breakout_level = coil_high * (1 + buf)
        # Previous bar broke out
        prev_broke = prev_close > breakout_level
        # Current bar retested (dipped into coil) and reclaimed
        retested = low <= coil_high * 1.001  # touched coil edge
        reclaimed = close > breakout_level
        return prev_broke and retested and reclaimed
    else:
        breakout_level = coil_low * (1 - buf)
        prev_broke = prev_close < breakout_level
        retested = high >= coil_low * 0.999
        reclaimed = close < breakout_level
        return prev_broke and retested and reclaimed


def entry_wick_conf(close, high, low, vol, vol_ma20, coil_high, coil_low, direction, cfg):
    """Breakout + strong wick rejection (body > 60% of range)."""
    buf = cfg['SQUEEZE_ENTRY_BUFFER_PCT']
    bar_range = high - low
    if bar_range <= 0:
        return False
    body = abs(close - (high + low) / 2)  # approximate body
    body_pct = body / bar_range
    if direction == 'LONG':
        return close > coil_high * (1 + buf) and body_pct > 0.60 and close > (high + low) / 2
    else:
        return close < coil_low * (1 - buf) and body_pct > 0.60 and close < (high + low) / 2


def entry_momentum(close, high, low, vol, vol_ma20, coil_high, coil_low, direction, rsi, cfg):
    """Breakout + RSI confirms direction (not overextended against)."""
    buf = cfg['SQUEEZE_ENTRY_BUFFER_PCT']
    if direction == 'LONG':
        return close > coil_high * (1 + buf) and rsi > 45 and rsi < 75
    else:
        return close < coil_low * (1 - buf) and rsi < 55 and rsi > 25


def entry_two_bar(closes, highs, lows, vols, vol_ma20s, coil_high, coil_low, direction, cfg, idx):
    """2 consecutive closes beyond coil range."""
    buf = cfg['SQUEEZE_ENTRY_BUFFER_PCT']
    if idx < 1:
        return False
    if direction == 'LONG':
        level = coil_high * (1 + buf)
        return closes[idx] > level and closes[idx-1] > level
    else:
        level = coil_low * (1 - buf)
        return closes[idx] < level and closes[idx-1] < level


def entry_vol_spike(close, high, low, vol, vol_ma20, coil_high, coil_low, direction, cfg):
    """Breakout + volume spike (≥ 1.5x MA20)."""
    buf = cfg['SQUEEZE_ENTRY_BUFFER_PCT']
    if direction == 'LONG':
        return close > coil_high * (1 + buf) and vol >= vol_ma20 * 1.5
    else:
        return close < coil_low * (1 - buf) and vol >= vol_ma20 * 1.5


def entry_combo(close, high, low, vol, vol_ma20, coil_high, coil_low, direction, rsi, cfg):
    """Wick confirmation + momentum + moderate volume."""
    buf = cfg['SQUEEZE_ENTRY_BUFFER_PCT']
    bar_range = high - low
    if bar_range <= 0:
        return False
    # Green/red candle in direction
    if direction == 'LONG':
        is_green = close > (high + low) / 2
        vol_ok = vol >= vol_ma20 * 1.2
        rsi_ok = 45 < rsi < 75
        return close > coil_high * (1 + buf) and is_green and vol_ok and rsi_ok
    else:
        is_red = close < (high + low) / 2
        vol_ok = vol >= vol_ma20 * 1.2
        rsi_ok = 25 < rsi < 55
        return close < coil_low * (1 - buf) and is_red and vol_ok and rsi_ok


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def compute_rsi(closes, period=14):
    """Compute RSI series."""
    delta = pd.Series(closes).diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).values


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['Open time'] = pd.to_datetime(df['Open time'])
    for c in ['Open', 'High', 'Low', 'Close', 'Volume',
              'Taker buy base asset volume', 'Taker buy quote asset volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['vol_ma20'] = df['Volume'].rolling(20).mean()
    df['taker_ratio'] = (df['Taker buy base asset volume'] / df['Volume'].replace(0, np.nan)).fillna(0.5)
    rsi_vals = compute_rsi(df['Close'].values)
    df['rsi'] = rsi_vals
    return df


def get_coil_range(df, coil_bars, idx):
    """Get 2h coil range from the coiled bars."""
    if coil_bars <= 0:
        return None, None
    # Approximate 2h bars as 8 x 15m bars
    start = max(0, idx - coil_bars * 8)
    h = float(df['High'].iloc[start:idx+1].max())
    l = float(df['Low'].iloc[start:idx+1].min())
    return h, l


def check_trade_outcome(closes, highs, lows, entry_price, direction, sl_pct, tp_pct, lookforward):
    """Check if trade hits TP or SL within lookforward bars."""
    if direction == 'LONG':
        sl_price = entry_price * (1 - sl_pct / 100)
        tp_price = entry_price * (1 + tp_pct / 100)
        for i in range(lookforward):
            if i >= len(closes):
                break
            if lows[i] <= sl_price:
                return 'SL', i, (sl_price - entry_price) / entry_price * 100
            if highs[i] >= tp_price:
                return 'TP', i, (tp_price - entry_price) / entry_price * 100
        # Neither hit — exit at last bar
        final = closes[min(lookforward-1, len(closes)-1)]
        return 'TIMEOUT', lookforward, (final - entry_price) / entry_price * 100
    else:
        sl_price = entry_price * (1 + sl_pct / 100)
        tp_price = entry_price * (1 - tp_pct / 100)
        for i in range(lookforward):
            if i >= len(closes):
                break
            if highs[i] >= sl_price:
                return 'SL', i, (entry_price - sl_price) / entry_price * 100
            if lows[i] <= tp_price:
                return 'TP', i, (entry_price - tp_price) / entry_price * 100
        final = closes[min(lookforward-1, len(closes)-1)]
        return 'TIMEOUT', lookforward, (entry_price - final) / entry_price * 100


# ═══════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ═══════════════════════════════════════════════════════════════

def run_backtest(csv_path, year=2026, lookforward=48, sl_pct=0.5, tp_pct=1.0):
    """Run backtest comparing all entry approaches."""
    df = load_data(csv_path)
    df_year = df[df['Open time'].dt.year == year].copy().reset_index(drop=True)

    cfg = SQUEEZE_V5_DEFAULTS
    snapshot_interval = 8  # check every 2h (8 x 15m)
    lookback = 48  # 12h lookback for compression

    approaches = {
        '1_BASELINE': [],
        '2_RETEST': [],
        '3_WICK_CONF': [],
        '4_MOMENTUM': [],
        '5_TWO_BAR': [],
        '6_VOL_SPIKE': [],
        '7_COMBO': [],
    }

    closes = df_year['Close'].values
    highs = df_year['High'].values
    lows = df_year['Low'].values
    vols = df_year['Volume'].values
    vol_ma20s = df_year['vol_ma20'].values
    rsis = df_year['rsi'].values

    idx = lookback
    last_squeeze_bar = -100

    while idx < len(df_year) - lookforward:
        # Skip if too close to last squeeze
        if idx - last_squeeze_bar < 32:
            idx += snapshot_interval
            continue

        # Check compression (Path A)
        compression_history = []
        for i in range(max(0, idx - 47), idx):
            r48_h = float(highs[max(0, i-47):i+1].max())
            r48_l = float(lows[max(0, i-47):i+1].min())
            r48_pct = (r48_h - r48_l) / float(closes[i]) * 100 if closes[i] > 0 else 5.0
            vr = float(vols[i] / vol_ma20s[i]) if vol_ma20s[i] > 0 else 1.0
            br = (float(highs[i]) - float(lows[i])) / float(closes[i]) * 100 if closes[i] > 0 else 0.5
            tr = float(df_year['taker_ratio'].iloc[i]) if 'taker_ratio' in df_year.columns else 0.5
            compression_history.append((r48_pct, vr, br, tr))

        r48_h = float(highs[max(0, idx-47):idx+1].max())
        r48_l = float(lows[max(0, idx-47):idx+1].min())
        range48 = (r48_h - r48_l) / float(closes[idx]) * 100

        a_compressed, a_comp_bars, a_dry, a_doji = _check_compression(range48, compression_history, cfg)

        # Check 2h MACD coil (Path B)
        b_coiled = False
        b_coil_bars = 0
        b_hist_flip = False
        b_direction = 'NEUTRAL'
        b_details = {}

        df_slice = df_year.iloc[:idx+1]
        macd_data = _compute_2h_macd(df_slice, cfg)
        if macd_data is not None:
            b_coiled, b_coil_bars, b_hist_flip, b_direction, b_details = \
                _detect_macd_coil(macd_data, cfg)

        # Need at least one path
        if not a_compressed and not b_coiled:
            idx += snapshot_interval
            continue

        # Resolve direction
        direction = 'NEUTRAL'
        if b_coiled and b_hist_flip:
            direction = b_direction
        elif a_compressed:
            taker = float(df_year['taker_ratio'].iloc[idx])
            if taker >= 0.58:
                direction = 'LONG'
            elif taker <= 0.42:
                direction = 'SHORT'

        if direction == 'NEUTRAL' and b_coiled and b_direction != 'NEUTRAL':
            direction = b_direction

        if direction == 'NEUTRAL':
            idx += snapshot_interval
            continue

        # Get coil range
        coil_high = b_details.get('coil_high', float(closes[idx]))
        coil_low = b_details.get('coil_low', float(closes[idx]))
        if a_compressed:
            a_start = max(0, idx - a_comp_bars)
            a_high = float(highs[a_start:idx+1].max())
            a_low = float(lows[a_start:idx+1].min())
            if (a_high - a_low) < (coil_high - coil_low):
                coil_high = a_high
                coil_low = a_low

        last_squeeze_bar = idx

        # Test each entry approach on the NEXT bars
        for bar_offset in range(1, min(lookforward, len(df_year) - idx)):
            bi = idx + bar_offset
            if bi >= len(df_year):
                break

            c = float(closes[bi])
            h = float(highs[bi])
            l = float(lows[bi])
            v = float(vols[bi])
            vm = float(vol_ma20s[bi]) if not np.isnan(vol_ma20s[bi]) else v
            r = float(rsis[bi]) if not np.isnan(rsis[bi]) else 50

            pc = float(closes[bi-1]) if bi > 0 else c
            ph = float(highs[bi-1]) if bi > 0 else h
            pl = float(lows[bi-1]) if bi > 0 else l

            # Check each approach
            triggered = {}

            if entry_baseline(c, h, l, v, vm, coil_high, coil_low, direction, cfg):
                triggered['1_BASELINE'] = bi

            if entry_retest(c, h, l, v, vm, coil_high, coil_low, direction, pc, ph, pl, cfg):
                triggered['2_RETEST'] = bi

            if entry_wick_conf(c, h, l, v, vm, coil_high, coil_low, direction, cfg):
                triggered['3_WICK_CONF'] = bi

            if entry_momentum(c, h, l, v, vm, coil_high, coil_low, direction, r, cfg):
                triggered['4_MOMENTUM'] = bi

            if entry_two_bar(closes, highs, lows, vols, vol_ma20s, coil_high, coil_low, direction, cfg, bi):
                triggered['5_TWO_BAR'] = bi

            if entry_vol_spike(c, h, l, v, vm, coil_high, coil_low, direction, cfg):
                triggered['6_VOL_SPIKE'] = bi

            if entry_combo(c, h, l, v, vm, coil_high, coil_low, direction, r, cfg):
                triggered['7_COMBO'] = bi

            # Record all triggered approaches (not just first bar)
            for approach, entry_bar in triggered.items():
                if any(t['squeeze_bar'] == idx for t in approaches[approach]):
                    continue  # already recorded for this squeeze

                entry_price = float(closes[entry_bar])
                future_closes = closes[entry_bar+1:entry_bar+1+lookforward]
                future_highs = highs[entry_bar+1:entry_bar+1+lookforward]
                future_lows = lows[entry_bar+1:entry_bar+1+lookforward]

                outcome, bars_held, pnl_pct = check_trade_outcome(
                    future_closes, future_highs, future_lows,
                    entry_price, direction, sl_pct, tp_pct, lookforward)

                approaches[approach].append({
                    'squeeze_bar': idx,
                    'entry_bar': entry_bar,
                    'delay_bars': entry_bar - idx,
                    'direction': direction,
                    'entry_price': entry_price,
                    'coil_high': coil_high,
                    'coil_low': coil_low,
                    'outcome': outcome,
                    'bars_held': bars_held,
                    'pnl_pct': pnl_pct,
                    'coil_hours': b_details.get('coil_hours', 0),
                })

        idx += snapshot_interval

    return approaches


def print_results(approaches, sl_pct, tp_pct):
    """Print comparison table."""
    print(f"\n{'='*85}")
    print(f"  SQUEEZE ENTRY APPROACH BACKTEST")
    print(f"  SL={sl_pct}%  TP={tp_pct}%  R:R={tp_pct/sl_pct:.1f}x")
    print(f"{'='*85}\n")

    header = f"{'Approach':<16} {'Trades':>6} {'Win%':>6} {'AvgPnL':>7} {'AvgWin':>7} {'AvgLoss':>8} {'AvgDelay':>8} {'Expect':>7}"
    print(header)
    print('-' * len(header))

    summary = []
    for name, trades in approaches.items():
        if not trades:
            print(f"  {name:<14} {'0':>6} {'—':>6} {'—':>7} {'—':>7} {'—':>8} {'—':>8} {'—':>7}")
            continue

        wins = [t for t in trades if t['outcome'] == 'TP']
        losses = [t for t in trades if t['outcome'] == 'SL']
        timeouts = [t for t in trades if t['outcome'] == 'TIMEOUT']

        n = len(trades)
        win_pct = len(wins) / n * 100 if n > 0 else 0
        avg_pnl = np.mean([t['pnl_pct'] for t in trades])
        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
        avg_delay = np.mean([t['delay_bars'] for t in trades])
        # Expectancy = win% * avg_win + loss% * avg_loss
        expectancy = (win_pct/100 * avg_win) + ((100-win_pct)/100 * avg_loss)

        print(f"  {name:<14} {n:>6} {win_pct:>5.1f}% {avg_pnl:>+6.2f}% {avg_win:>+6.2f}% {avg_loss:>+7.2f}% {avg_delay:>7.1f}b {expectancy:>+6.3f}%")

        summary.append({
            'name': name, 'trades': n, 'win_pct': win_pct,
            'avg_pnl': avg_pnl, 'expectancy': expectancy,
            'avg_delay': avg_delay,
        })

    # Find best
    if summary:
        best_wr = max(summary, key=lambda x: x['win_pct'])
        best_exp = max(summary, key=lambda x: x['expectancy'])
        fastest = min(summary, key=lambda x: x['avg_delay'])

        print(f"\n  🏆 Best win rate:    {best_wr['name']} ({best_wr['win_pct']:.1f}%)")
        print(f"  📈 Best expectancy:  {best_exp['name']} ({best_exp['expectancy']:+.3f}%)")
        print(f"  ⚡ Fastest entry:    {fastest['name']} ({fastest['avg_delay']:.1f} bars delay)")

    # Coil duration breakdown
    print(f"\n  Coil Duration Breakdown (BASELINE):")
    baseline = approaches.get('1_BASELINE', [])
    if baseline:
        short_coils = [t for t in baseline if t['coil_hours'] < 18]
        long_coils = [t for t in baseline if t['coil_hours'] >= 18]
        for label, subset in [('<18h', short_coils), ('≥18h', long_coils)]:
            if not subset:
                continue
            wins = [t for t in subset if t['outcome'] == 'TP']
            n = len(subset)
            wr = len(wins) / n * 100 if n > 0 else 0
            avg_pnl = np.mean([t['pnl_pct'] for t in subset])
            print(f"    {label}: {n} trades, WR={wr:.1f}%, avg={avg_pnl:+.2f}%")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2026)
    parser.add_argument('--lookforward', type=int, default=48)
    parser.add_argument('--sl', type=float, default=0.5)
    parser.add_argument('--tp', type=float, default=1.0)
    parser.add_argument('--csv', default='eth_15m_merged.csv')
    args = parser.parse_args()

    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.csv)
    approaches = run_backtest(csv_path, args.year, args.lookforward, args.sl, args.tp)
    print_results(approaches, args.sl, args.tp)
