#!/usr/bin/env python3
"""
Backtest: M-Signal (technical composite) vs M22 (macro regime)
Tests whether the new 3-dimension composite predicts 24h returns better than M22.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import sys, os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

from backtest_m22_proposal import (
    PPI_RELEASES, CPI_RELEASES, NFP_RELEASES,
    load_eth_data, compute_return, m22_lookup_score,
    get_claims_for_date, get_fed_rate,
)
from src.utils.indicators import calc_macd


def compute_range_pct(high, low, close, idx, lookback=48):
    roll_high = np.max(high[max(0, idx-lookback+1):idx+1])
    roll_low = np.min(low[max(0, idx-lookback+1):idx+1])
    return (roll_high - roll_low) / close[idx] * 100


def compute_dist_from_high(high, close, idx, lookback=48):
    roll_high = np.max(high[max(0, idx-lookback+1):idx+1])
    return (close[idx] - roll_high) / roll_high * 100


def compute_macd_hist(close, idx, fast=12, slow=26, signal=9):
    lookback = min(idx + 1, 600)
    c = pd.Series(close[max(0, idx-lookback+1):idx+1].astype(float))
    _, _, hist = calc_macd(c, fast, slow, signal)
    return float(hist.iloc[-1]) if not np.isnan(hist.iloc[-1]) else 0


def signal_score(range_pct, dist_from_high, macd_hist, atr_pct, direction='LONG'):
    """Compute M-Signal composite score."""
    # Normalize each dimension
    range_score = max(0.0, min(1.0, (range_pct - 0.5) / 4.5))
    dist_score = max(0.0, min(1.0, (-dist_from_high - 0.5) / 4.5))
    hist_normalized = macd_hist / (atr_pct * 100) if atr_pct > 0 else 0
    hist_score = max(0.0, min(1.0, (hist_normalized + 2) / 4))

    raw = range_score * 0.36 + dist_score * 0.33 + hist_score * 0.31

    if direction == 'SHORT':
        return 1.0 - raw
    return raw


def main():
    df_15m = load_eth_data('eth_15m_merged.csv')
    df_15m['Open time'] = pd.to_datetime(df_15m['Open time'])
    close = df_15m['Close'].values.astype(float)
    high = df_15m['High'].values.astype(float)
    low = df_15m['Low'].values.astype(float)

    # Build releases
    all_dates = sorted(set(list(PPI_RELEASES.keys()) + list(CPI_RELEASES.keys()) + list(NFP_RELEASES.keys())))

    results = []
    for date_str in all_dates:
        ret_24h = compute_return(df_15m, date_str, hours=24)
        if ret_24h is None:
            continue

        release_dt = pd.Timestamp(f"{date_str} 13:30:00")
        mask = df_15m['Open time'] <= release_dt
        if mask.sum() == 0:
            continue
        idx = mask.sum() - 1
        if idx < 60:
            continue

        price = close[idx]

        # Technical dimensions
        range_pct = compute_range_pct(high, low, close, idx)
        dist_from_high = compute_dist_from_high(high, close, idx)
        macd_hist = compute_macd_hist(close, idx)

        # ATR for normalization
        from src.utils.indicators import calc_atr
        atr_s = calc_atr(pd.Series(high[max(0,idx-59):idx+1]),
                          pd.Series(low[max(0,idx-59):idx+1]),
                          pd.Series(close[max(0,idx-59):idx+1]), 14)
        atr_val = float(atr_s.iloc[-1]) if not np.isnan(atr_s.iloc[-1]) else price * 0.02
        atr_pct = atr_val / price

        # M-Signal score (LONG direction — we test both directions later)
        sig_long = signal_score(range_pct, dist_from_high, macd_hist, atr_pct, 'LONG')
        sig_short = signal_score(range_pct, dist_from_high, macd_hist, atr_pct, 'SHORT')

        # M22 macro score
        claims = get_claims_for_date(date_str)
        fed_rate = get_fed_rate(date_str)
        # Get PPI/CPI context
        ppi_yoy, ppi_prior = 2.0, 2.0
        cpi_yoy = 2.0
        for d in sorted(PPI_RELEASES.keys(), reverse=True):
            if d <= date_str:
                ppi_yoy = PPI_RELEASES[d]['ppi_yoy']
                ppi_prior = PPI_RELEASES[d]['prior_yoy']
                break
        for d in sorted(CPI_RELEASES.keys(), reverse=True):
            if d <= date_str:
                cpi_yoy = CPI_RELEASES[d]['cpi_yoy']
                break

        m22 = m22_lookup_score(ppi_yoy, ppi_prior, cpi_yoy, fed_rate, claims)

        # NFP surprise
        nfp_surprise = 0
        if date_str in NFP_RELEASES:
            nfp_surprise = NFP_RELEASES[date_str]['nfp_k'] - NFP_RELEASES[date_str]['consensus_k']

        # Signal + NFP gate
        sig_long_nfp = sig_long
        if abs(nfp_surprise) > 50:
            sig_long_nfp = min(1.0, sig_long + 0.05)

        results.append({
            'date': date_str,
            'return_24h': ret_24h,
            'signal_long': sig_long,
            'signal_short': sig_short,
            'signal_long_nfp': sig_long_nfp,
            'm22': m22,
            'range_pct': range_pct,
            'dist_from_high': dist_from_high,
            'macd_hist': macd_hist,
            'nfp_surprise': nfp_surprise,
        })

    data = pd.DataFrame(results)
    print(f"Dataset: {len(data)} releases, {data['date'].min()} → {data['date'].max()}")

    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD: M-Signal vs M22")
    print("=" * 70)

    models = {
        'M22 (macro regime)': 'm22',
        'M-Signal (technical)': 'signal_long',
        'M-Signal + NFP gate': 'signal_long_nfp',
    }

    print(f"\n  {'Model':<25} {'Accuracy':>8} {'r':>8} {'p':>8} {'OOS Acc':>8} {'OOS r':>8}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for name, col in models.items():
        # Direction accuracy: score > 0.5 predicts positive return
        pred = data[col] > 0.5
        act = data['return_24h'] > 0
        acc = (pred == act).mean()

        corr, p_val = stats.pearsonr(data[col], data['return_24h'])

        # OOS
        oos = data[data['date'] >= '2024-01-01']
        if len(oos) > 10:
            oos_pred = oos[col] > 0.5
            oos_act = oos['return_24h'] > 0
            oos_acc = (oos_pred == oos_act).mean()
            oos_corr, _ = stats.pearsonr(oos[col], oos['return_24h'])
        else:
            oos_acc, oos_corr = 0, 0

        print(f"  {name:<25} {acc:>7.1%} {corr:>+7.3f} {p_val:>7.4f} {oos_acc:>7.1%} {oos_corr:>+7.3f}")

    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  DIRECTIONALITY TEST — Does the composite pick the right side?")
    print("=" * 70)

    for name, col in [('M22', 'm22'), ('M-Signal', 'signal_long'), ('M-Signal+NFP', 'signal_long_nfp')]:
        # When score > 0.55 (bullish), what's the avg return?
        bullish = data[data[col] > 0.55]
        bearish = data[data[col] < 0.45]
        neutral = data[(data[col] >= 0.45) & (data[col] <= 0.55)]

        print(f"\n  {name}:")
        if len(bullish) > 0:
            print(f"    Bullish (>0.55): avg={bullish['return_24h'].mean():+.2f}%  n={len(bullish)}  "
                  f"win={((bullish['return_24h'] > 0).mean()):.0%}")
        if len(neutral) > 0:
            print(f"    Neutral (0.45-0.55): avg={neutral['return_24h'].mean():+.2f}%  n={len(neutral)}")
        if len(bearish) > 0:
            print(f"    Bearish (<0.45): avg={bearish['return_24h'].mean():+.2f}%  n={len(bearish)}  "
                  f"win={((bearish['return_24h'] < 0).mean()):.0%}")

        spread = 0
        if len(bullish) > 0 and len(bearish) > 0:
            spread = bullish['return_24h'].mean() - bearish['return_24h'].mean()
        print(f"    Bull-Bear spread: {spread:+.2f}%")

    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  HIGH-CONFIDENCE SIGNALS — Score > 0.65 or < 0.35")
    print("=" * 70)

    for name, col in [('M22', 'm22'), ('M-Signal', 'signal_long'), ('M-Signal+NFP', 'signal_long_nfp')]:
        hc = data[(data[col] > 0.65) | (data[col] < 0.35)]
        if len(hc) == 0:
            print(f"\n  {name}: no high-confidence signals")
            continue

        hc_pred = hc[col] > 0.5
        hc_act = hc['return_24h'] > 0
        hc_acc = (hc_pred == hc_act).mean()

        # Win rate when signal matches actual direction
        bull_win = hc[(hc[col] > 0.65) & (hc['return_24h'] > 0)]
        bear_win = hc[(hc[col] < 0.35) & (hc['return_24h'] < 0)]

        print(f"\n  {name} (n={len(hc)}):")
        print(f"    Direction accuracy: {hc_acc:.1%}")
        print(f"    Avg return: {hc['return_24h'].mean():+.2f}%")
        print(f"    Bullish signals win: {((hc[hc[col] > 0.65]['return_24h'] > 0).mean()):.0%}  "
              f"(n={len(hc[hc[col] > 0.65])})")
        print(f"    Bearish signals win: {((hc[hc[col] < 0.35]['return_24h'] < 0).mean()):.0%}  "
              f"(n={len(hc[hc[col] < 0.35])})")

    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  OOS BREAKDOWN (2024-2026)")
    print("=" * 70)

    oos = data[data['date'] >= '2024-01-01']
    for name, col in [('M22', 'm22'), ('M-Signal+NFP', 'signal_long_nfp')]:
        pred = oos[col] > 0.5
        act = oos['return_24h'] > 0
        acc = (pred == act).mean()
        corr, p_val = stats.pearsonr(oos[col], oos['return_24h'])

        bull = oos[oos[col] > 0.55]
        bear = oos[oos[col] < 0.45]

        print(f"\n  {name}:")
        print(f"    Accuracy: {acc:.1%}  (n={len(oos)})")
        print(f"    Correlation: r={corr:+.3f}  p={p_val:.4f}")
        if len(bull) > 0:
            print(f"    Bullish signals: avg={bull['return_24h'].mean():+.2f}%  win={((bull['return_24h'] > 0).mean()):.0%}")
        if len(bear) > 0:
            print(f"    Bearish signals: avg={bear['return_24h'].mean():+.2f}%  win={((bear['return_24h'] < 0).mean()):.0%}")

    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  ICS IMPACT — What does this do to the signal pipeline?")
    print("=" * 70)

    # If M-Signal replaces M22 in ICS, what changes?
    # M22 currently contributes ~15% weight to ICS (one of ~12 modules)
    # M-Signal would contribute the same weight but with better predictive power

    m22_above = (data['m22'] > 0.5).sum()
    sig_above = (data['signal_long_nfp'] > 0.5).sum()
    agree = ((data['m22'] > 0.5) == (data['signal_long_nfp'] > 0.5)).sum()

    print(f"\n  M22 vs M-Signal agreement: {agree}/{len(data)} ({agree/len(data):.0%})")
    print(f"  M22 bullish: {m22_above}/{len(data)} ({m22_above/len(data):.0%})")
    print(f"  M-Signal bullish: {sig_above}/{len(data)} ({sig_above/len(data):.0%})")

    # When they disagree, who's right?
    disagree = data[(data['m22'] > 0.5) != (data['signal_long_nfp'] > 0.5)]
    if len(disagree) > 0:
        m22_right = ((disagree['m22'] > 0.5) == (disagree['return_24h'] > 0)).mean()
        sig_right = ((disagree['signal_long_nfp'] > 0.5) == (disagree['return_24h'] > 0)).mean()
        print(f"\n  When they disagree (n={len(disagree)}):")
        print(f"    M22 correct: {m22_right:.0%}")
        print(f"    M-Signal correct: {sig_right:.0%}")


if __name__ == '__main__':
    main()
