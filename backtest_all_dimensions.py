#!/usr/bin/env python3
"""
Backtest: ALL module dimensions vs 24h ETH return.

Tests every computable dimension from M1-M62 to find the best predictor set.
Goal: find which modules carry signal and which are noise.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS (computed from ETH 15m data)
# ═══════════════════════════════════════════════════════════════

def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_vwap(high, low, close, volume, lookback=20):
    typical = (high + low + close) / 3
    cum_tp_vol = (typical * volume).rolling(lookback).sum()
    cum_vol = volume.rolling(lookback).sum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)

def resample_1h(df):
    """Resample 15m to 1h."""
    df = df.copy()
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.set_index('Open time')
    agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    resampled = df.resample('1h').agg(agg).dropna(subset=['Open'])
    return resampled.reset_index()

def resample_4h(df):
    """Resample 15m to 4h."""
    df = df.copy()
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.set_index('Open time')
    agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    resampled = df.resample('4h').agg(agg).dropna(subset=['Open'])
    return resampled.reset_index()

def resample_1d(df):
    """Resample 15m to daily."""
    df = df.copy()
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.set_index('Open time')
    agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    resampled = df.resample('1D').agg(agg).dropna(subset=['Open'])
    return resampled.reset_index()


# ═══════════════════════════════════════════════════════════════
# DIMENSION COMPUTERS
# ═══════════════════════════════════════════════════════════════

def compute_technical_features(df_15m, idx):
    """Compute technical indicator dimensions at a given bar index."""
    if idx < 100:
        return None

    close = df_15m['Close'].values.astype(float)
    high = df_15m['High'].values.astype(float)
    low = df_15m['Low'].values.astype(float)
    volume = df_15m['Volume'].values.astype(float)
    price = close[idx]

    features = {}

    # ── M1: MACD (1H resampled) ──
    # Use last 100 bars for 1H MACD approximation
    lookback = min(idx + 1, 600)
    c = pd.Series(close[max(0, idx-lookback+1):idx+1])
    macd_line, macd_signal, macd_hist = calc_macd(c, 12, 26, 9)
    features['macd_hist'] = float(macd_hist.iloc[-1]) if not pd.isna(macd_hist.iloc[-1]) else 0
    features['macd_cross'] = 1 if float(macd_line.iloc[-1]) > float(macd_signal.iloc[-1]) else -1

    # ── M2: EMA confluence ──
    ema21 = calc_ema(c, 21).iloc[-1]
    ema55 = calc_ema(c, 55).iloc[-1]
    features['ema_spread'] = (float(ema21) - float(ema55)) / float(ema55) * 100
    features['ema_trend'] = 1 if ema21 > ema55 else -1

    # Price vs EMAs
    features['price_vs_ema21'] = (price - float(ema21)) / float(ema21) * 100
    features['price_vs_ema55'] = (price - float(ema55)) / float(ema55) * 100

    # ── M3: VWAP ──
    h = pd.Series(high[max(0, idx-19):idx+1])
    l = pd.Series(low[max(0, idx-19):idx+1])
    cl = pd.Series(close[max(0, idx-19):idx+1])
    v = pd.Series(volume[max(0, idx-19):idx+1])
    vwap = calc_vwap(h, l, cl, v, 20)
    vwap_val = float(vwap.iloc[-1]) if not pd.isna(vwap.iloc[-1]) else price
    features['vwap_dist'] = (price - vwap_val) / vwap_val * 100

    # ── M4: CVD proxy (taker ratio) ──
    if 'Taker buy base asset volume' in df_15m.columns:
        taker = df_15m['Taker buy base asset volume'].values.astype(float)
        total = volume
        taker_ratio = taker / np.where(total > 0, total, 1)
        # 20-bar rolling average
        avg_taker = np.mean(taker_ratio[max(0, idx-19):idx+1])
        features['taker_ratio'] = float(avg_taker)
        features['taker_bias'] = 1 if avg_taker > 0.52 else -1 if avg_taker < 0.48 else 0
    else:
        features['taker_ratio'] = 0.5
        features['taker_bias'] = 0

    # ── M9: Volatility regime ──
    atr14 = calc_atr(pd.Series(high[max(0,idx-59):idx+1]),
                      pd.Series(low[max(0,idx-59):idx+1]),
                      pd.Series(close[max(0,idx-59):idx+1]), 14)
    atr_val = float(atr14.iloc[-1]) if not pd.isna(atr14.iloc[-1]) else 0
    features['atr'] = atr_val
    features['atr_pct'] = atr_val / price * 100

    # Vol expansion/contraction
    atr_20 = atr14.rolling(20).mean()
    if not pd.isna(atr_20.iloc[-1]) and atr_20.iloc[-1] > 0:
        features['vol_regime'] = float(atr14.iloc[-1] / atr_20.iloc[-1])
    else:
        features['vol_regime'] = 1.0

    # ── M13: Structure (swing bias) ──
    # Higher highs / higher lows over last 48 bars
    lookback_48 = min(48, idx)
    recent_highs = high[idx-lookback_48:idx+1]
    recent_lows = low[idx-lookback_48:idx+1]
    swing_high_1 = np.max(recent_highs[:lookback_48//2])
    swing_high_2 = np.max(recent_highs[lookback_48//2:])
    swing_low_1 = np.min(recent_lows[:lookback_48//2])
    swing_low_2 = np.min(recent_lows[lookback_48//2:])

    if swing_high_2 > swing_high_1 and swing_low_2 > swing_low_1:
        features['swing_bias'] = 1  # bullish
    elif swing_high_2 < swing_high_1 and swing_low_2 < swing_low_1:
        features['swing_bias'] = -1  # bearish
    else:
        features['swing_bias'] = 0  # neutral

    # ── RSI ──
    rsi = calc_rsi(pd.Series(close[max(0,idx-59):idx+1]), 14)
    features['rsi'] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
    features['rsi_extreme'] = 1 if features['rsi'] > 70 else -1 if features['rsi'] < 30 else 0

    # ── M9b: Range width (squeeze proxy) ──
    roll_high = np.max(high[max(0,idx-47):idx+1])
    roll_low = np.min(low[max(0,idx-47):idx+1])
    features['range_pct'] = (roll_high - roll_low) / price * 100

    # ── Volume trend ──
    vol_20 = np.mean(volume[max(0,idx-19):idx+1])
    vol_60 = np.mean(volume[max(0,idx-59):idx+1])
    features['vol_ratio'] = vol_20 / vol_60 if vol_60 > 0 else 1.0

    # ── M5: Liquidation proxy (distance from recent high/low) ──
    dist_from_high = (price - roll_high) / roll_high * 100
    dist_from_low = (price - roll_low) / roll_low * 100
    features['dist_from_high'] = dist_from_high
    features['dist_from_low'] = dist_from_low

    # ── M21: Wyckoff phase proxy ──
    # Simple: accumulation (low vol + range), markup (rising), distribution (high vol + range), markdown (falling)
    if features['vol_regime'] < 0.8 and abs(features['swing_bias']) == 0:
        features['wyckoff_phase'] = 0  # accumulation/range
    elif features['swing_bias'] == 1 and features['vol_regime'] > 1.0:
        features['wyckoff_phase'] = 1  # markup
    elif features['vol_regime'] < 0.8 and abs(features['swing_bias']) == 0 and features['range_pct'] > 3:
        features['wyckoff_phase'] = 2  # distribution
    elif features['swing_bias'] == -1:
        features['wyckoff_phase'] = -1  # markdown
    else:
        features['wyckoff_phase'] = 0

    # ── Daily trend (from 1d resample) ──
    # Use 96 bars = 1 day on 15m
    if idx >= 96:
        daily_close = close[idx-96]
        features['return_1d'] = (price - daily_close) / daily_close * 100
    else:
        features['return_1d'] = 0

    # ── Weekly trend ──
    if idx >= 480:
        weekly_close = close[idx-480]
        features['return_7d'] = (price - weekly_close) / weekly_close * 100
    else:
        features['return_7d'] = 0

    # ── Momentum ──
    if idx >= 16:
        features['return_4h'] = (price - close[idx-16]) / close[idx-16] * 100
    else:
        features['return_4h'] = 0

    return features


def compute_macro_features(date_str, ppi_data, cpi_data, nfp_data):
    """Compute macro dimension features at a release date."""
    claims_map = {
        '2018-01': 230, '2018-06': 220, '2018-12': 215,
        '2019-01': 210, '2019-06': 215, '2019-12': 220,
        '2020-01': 210, '2020-03': 300, '2020-06': 1500, '2020-12': 800,
        '2021-01': 900, '2021-06': 400, '2021-12': 200,
        '2022-01': 210, '2022-06': 195, '2022-12': 195,
        '2023-01': 190, '2023-06': 215, '2023-12': 210,
        '2024-01': 210, '2024-06': 225, '2024-12': 215,
        '2025-01': 205, '2025-06': 230, '2025-12': 199,
        '2026-01': 210, '2026-04': 200, '2026-05': 200,
    }
    fed_map = {
        '2018-01': 1.42, '2018-06': 1.92, '2018-12': 2.40,
        '2019-01': 2.40, '2019-06': 2.40, '2019-12': 1.55,
        '2020-01': 1.55, '2020-06': 0.05, '2020-12': 0.05,
        '2021-01': 0.05, '2021-06': 0.05, '2021-12': 0.08,
        '2022-01': 0.08, '2022-06': 1.21, '2022-12': 4.33,
        '2023-01': 4.50, '2023-06': 5.08, '2023-12': 5.33,
        '2024-01': 5.33, '2024-06': 5.33, '2024-12': 4.33,
        '2025-01': 4.33, '2025-06': 4.33, '2025-12': 4.33,
        '2026-01': 4.33, '2026-04': 4.33, '2026-05': 4.33,
    }

    def lookup_approx(d, date_str):
        y, m = int(date_str[:4]), int(date_str[5:7])
        for offset in range(0, 6):
            key = f"{y:04d}-{m-offset:02d}" if m - offset > 0 else f"{y-1:04d}-{12+(m-offset):02d}"
            if key in d:
                return d[key]
        return list(d.values())[0]

    def latest_before(date_str, data):
        for d in sorted(data.keys(), reverse=True):
            if d < date_str:
                return data[d]
        return None

    features = {}

    # PPI
    ppi = ppi_data.get(date_str) or latest_before(date_str, ppi_data)
    if ppi:
        features['ppi_yoy'] = ppi['ppi_yoy']
        features['ppi_surprise'] = ppi['ppi_yoy'] - ppi['consensus_yoy']
        features['ppi_delta'] = ppi['ppi_yoy'] - ppi['prior_yoy']
    else:
        features['ppi_yoy'] = 2.0
        features['ppi_surprise'] = 0
        features['ppi_delta'] = 0

    # CPI
    cpi = cpi_data.get(date_str) or latest_before(date_str, cpi_data)
    if cpi:
        features['cpi_yoy'] = cpi['cpi_yoy']
        features['cpi_surprise'] = cpi['cpi_yoy'] - cpi['consensus_yoy']
    else:
        features['cpi_yoy'] = 2.0
        features['cpi_surprise'] = 0

    # NFP
    nfp = nfp_data.get(date_str)
    if nfp:
        features['nfp_surprise_k'] = nfp['nfp_k'] - nfp['consensus_k']
    else:
        features['nfp_surprise_k'] = 0

    # Claims
    features['claims'] = lookup_approx(claims_map, date_str)

    # Fed rate
    features['fed_rate'] = lookup_approx(fed_map, date_str)

    # Real rate
    features['real_rate'] = features['fed_rate'] - features['ppi_yoy']

    # Inflation avg
    features['inflation_avg'] = (features['ppi_yoy'] + features['cpi_yoy']) / 2

    # Release type
    if date_str in ppi_data:
        features['is_ppi'] = 1
    else:
        features['is_ppi'] = 0
    if date_str in cpi_data:
        features['is_cpi'] = 1
    else:
        features['is_cpi'] = 0
    if date_str in nfp_data:
        features['is_nfp'] = 1
    else:
        features['is_nfp'] = 0

    return features


# ═══════════════════════════════════════════════════════════════
# RELEASE DATA
# ═══════════════════════════════════════════════════════════════

from backtest_m22_proposal import (
    PPI_RELEASES, CPI_RELEASES, NFP_RELEASES,
    load_eth_data, compute_return,
)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    df_15m = load_eth_data('eth_15m_merged.csv')
    df_15m['Open time'] = pd.to_datetime(df_15m['Open time'])

    # Build all release dates
    all_dates = sorted(set(list(PPI_RELEASES.keys()) + list(CPI_RELEASES.keys()) + list(NFP_RELEASES.keys())))

    # Pre-compute 1H resampled data for MACD
    print("Resampling to 1H...")
    df_1h = resample_1h(df_15m)

    results = []
    for date_str in all_dates:
        ret_24h = compute_return(df_15m, date_str, hours=24)
        if ret_24h is None:
            continue

        # Find the 15m bar index at release time (13:30 UTC)
        release_dt = pd.Timestamp(f"{date_str} 13:30:00")
        mask = df_15m['Open time'] <= release_dt
        if mask.sum() == 0:
            continue
        idx = mask.sum() - 1

        # Technical features
        tech = compute_technical_features(df_15m, idx)
        if tech is None:
            continue

        # Macro features
        macro = compute_macro_features(date_str, PPI_RELEASES, CPI_RELEASES, NFP_RELEASES)

        # Merge
        row = {'date': date_str, 'return_24h': ret_24h}
        row.update(tech)
        row.update(macro)
        results.append(row)

    data = pd.DataFrame(results)
    print(f"\nDataset: {len(data)} releases, {data['date'].min()} → {data['date'].max()}")

    # ═══════════════════════════════════════════════════════════════
    # ALL DIMENSIONS TEST
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  ALL DIMENSIONS — Pearson + Spearman correlation with 24h return")
    print("=" * 70)

    # All numeric columns except date and return
    skip_cols = {'date', 'return_24h'}
    dim_cols = [c for c in data.columns if c not in skip_cols and data[c].dtype in ('float64', 'int64', 'float32', 'int32')]

    results_list = []
    for col in dim_cols:
        subset = data[[col, 'return_24h']].dropna()
        if len(subset) < 20:
            continue
        r, p = stats.pearsonr(subset[col], subset['return_24h'])
        rs, ps = stats.spearmanr(subset[col], subset['return_24h'])
        results_list.append({
            'dim': col, 'r': r, 'p': p, 'rs': rs, 'ps': ps,
            'abs_r': abs(r), 'abs_rs': abs(rs),
        })

    df_corr = pd.DataFrame(results_list).sort_values('p')

    print(f"\n  {'Dimension':<25} {'Pearson':>8} {'p':>8} {'Spearman':>8} {'p':>8} {'Verdict'}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")
    for _, row in df_corr.iterrows():
        verdict = '✅' if row['p'] < 0.10 else '⚠️' if row['p'] < 0.25 else '❌'
        spear_sig = '*' if row['ps'] < 0.10 else ''
        print(f"  {row['dim']:<25} {row['r']:>+7.3f} {row['p']:>7.4f} {row['rs']:>+7.3f} {row['ps']:>7.4f}{spear_sig}  {verdict}")

    # ═══════════════════════════════════════════════════════════════
    # STEPWISE REGRESSION — find best N-dimension model
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  STEPWISE MODEL — Add dimensions by predictive power")
    print("=" * 70)

    # Sort by p-value
    ranked = df_corr.sort_values('p').reset_index(drop=True)

    used = []
    prev_r2 = 0

    print(f"\n  {'Step':<5} {'Add':<25} {'r':>8} {'R²':>8} {'ΔR²':>8} {'OOS r':>8}")
    print(f"  {'─'*5} {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for i, row in ranked.iterrows():
        if i >= 12:  # limit to top 12
            break
        col = row['dim']
        used.append(col)

        X = data[used].values
        y = data['return_24h'].values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X, y = X[mask], y[mask]

        if len(X) < 20:
            continue

        X_i = np.column_stack([np.ones(len(X)), X])
        beta, _, _, _ = np.linalg.lstsq(X_i, y, rcond=None)
        y_pred = X_i @ beta
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res / ss_tot
        corr = np.corrcoef(y_pred, y)[0, 1]
        delta_r2 = r2 - prev_r2

        # OOS
        train_mask = data.loc[mask, 'date'].values < '2024-01-01'
        test_mask = data.loc[mask, 'date'].values >= '2024-01-01'
        if test_mask.sum() > 10:
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            X_train_i = np.column_stack([np.ones(len(X_train)), X_train])
            X_test_i = np.column_stack([np.ones(len(X_test)), X_test])
            beta_train, _, _, _ = np.linalg.lstsq(X_train_i, y_train, rcond=None)
            y_pred_test = X_test_i @ beta_train
            oos_corr = np.corrcoef(y_pred_test, y_test)[0, 1]
        else:
            oos_corr = 0

        sig = '✅' if delta_r2 > 0.01 else ''
        print(f"  {len(used):<5} {col:<25} {corr:>+7.3f} {r2:>7.4f} {delta_r2:>+7.4f} {oos_corr:>+7.3f} {sig}")
        prev_r2 = r2

    # ═══════════════════════════════════════════════════════════════
    # BEST 3-DIMENSION COMBINATIONS
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  BEST 3-DIMENSION COMBINATIONS (brute force top 10 dims)")
    print("=" * 70)

    from itertools import combinations

    top_dims = list(ranked.head(10)['dim'])
    best_combos = []

    for combo in combinations(top_dims, 3):
        X = data[list(combo)].values
        y = data['return_24h'].values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_c, y_c = X[mask], y[mask]
        if len(X_c) < 30:
            continue

        # In-sample
        X_i = np.column_stack([np.ones(len(X_c)), X_c])
        beta, _, _, _ = np.linalg.lstsq(X_i, y_c, rcond=None)
        y_pred = X_i @ beta
        r2 = 1 - np.sum((y_c - y_pred)**2) / np.sum((y_c - y_c.mean())**2)
        corr = np.corrcoef(y_pred, y_c)[0, 1]

        # OOS
        train_mask = data.loc[mask, 'date'].values < '2024-01-01'
        test_mask = data.loc[mask, 'date'].values >= '2024-01-01'
        if test_mask.sum() > 10:
            X_train, y_train = X_c[train_mask], y_c[train_mask]
            X_test, y_test = X_c[test_mask], y_c[test_mask]
            X_ti = np.column_stack([np.ones(len(X_train)), X_train])
            X_tei = np.column_stack([np.ones(len(X_test)), X_test])
            beta_t, _, _, _ = np.linalg.lstsq(X_ti, y_train, rcond=None)
            y_pred_oos = X_tei @ beta_t
            oos_corr = np.corrcoef(y_pred_oos, y_test)[0, 1]
            oos_r2 = 1 - np.sum((y_test - y_pred_oos)**2) / np.sum((y_test - y_test.mean())**2)
        else:
            oos_corr, oos_r2 = 0, 0

        best_combos.append({
            'combo': combo, 'r': corr, 'r2': r2,
            'oos_r': oos_corr, 'oos_r2': oos_r2,
        })

    best_combos.sort(key=lambda x: abs(x['oos_r']), reverse=True)

    print(f"\n  {'Rank':<5} {'Combo':<50} {'IS r':>7} {'IS R²':>7} {'OOS r':>7} {'OOS R²':>7}")
    print(f"  {'─'*5} {'─'*50} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    for i, bc in enumerate(best_combos[:10]):
        combo_str = ' × '.join(bc['combo'])
        print(f"  {i+1:<5} {combo_str:<50} {bc['r']:>+6.3f} {bc['r2']:>6.4f} {bc['oos_r']:>+6.3f} {bc['oos_r2']:>6.4f}")

    # ═══════════════════════════════════════════════════════════════
    # CATEGORY ANALYSIS — which module category wins?
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  CATEGORY ANALYSIS — Technical vs Macro vs Structure")
    print("=" * 70)

    categories = {
        'Technical (M1-M4)': ['macd_hist', 'macd_cross', 'ema_spread', 'ema_trend',
                               'price_vs_ema21', 'vwap_dist', 'taker_ratio', 'rsi'],
        'Volatility (M9)': ['atr', 'atr_pct', 'vol_regime', 'range_pct', 'vol_ratio'],
        'Structure (M13/M21)': ['swing_bias', 'wyckoff_phase', 'dist_from_high', 'dist_from_low'],
        'Momentum': ['return_4h', 'return_1d', 'return_7d'],
        'Macro (M22)': ['ppi_yoy', 'ppi_surprise', 'ppi_delta', 'cpi_yoy', 'cpi_surprise',
                         'inflation_avg', 'real_rate', 'fed_rate', 'claims'],
    }

    for cat_name, cat_dims in categories.items():
        available = [d for d in cat_dims if d in data.columns and d in dim_cols]
        if not available:
            continue

        # Test each
        sig_count = 0
        best_r = 0
        best_dim = ''
        for d in available:
            subset = data[[d, 'return_24h']].dropna()
            if len(subset) < 20:
                continue
            r, p = stats.pearsonr(subset[d], subset['return_24h'])
            if p < 0.10:
                sig_count += 1
            if abs(r) > abs(best_r):
                best_r = r
                best_dim = d

        # Combined
        X = data[available].values
        y = data['return_24h'].values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_c, y_c = X[mask], y[mask]
        if len(X_c) > 20:
            X_i = np.column_stack([np.ones(len(X_c)), X_c])
            beta, _, _, _ = np.linalg.lstsq(X_i, y_c, rcond=None)
            y_pred = X_i @ beta
            combined_r = np.corrcoef(y_pred, y_c)[0, 1]
            combined_r2 = 1 - np.sum((y_c - y_pred)**2) / np.sum((y_c - y_c.mean())**2)
        else:
            combined_r, combined_r2 = 0, 0

        print(f"\n  {cat_name}:")
        print(f"    Significant dims: {sig_count}/{len(available)}")
        print(f"    Best single: {best_dim} (r={best_r:+.3f})")
        print(f"    Combined: r={combined_r:+.3f}  R²={combined_r2:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    sig_dims = df_corr[df_corr['p'] < 0.10]
    weak_dims = df_corr[(df_corr['p'] >= 0.10) & (df_corr['p'] < 0.25)]
    noise_dims = df_corr[df_corr['p'] >= 0.25]

    print(f"\n  ✅ Significant (p<0.10): {len(sig_dims)}")
    for _, r in sig_dims.iterrows():
        print(f"    {r['dim']:<25} r={r['r']:+.3f}  p={r['p']:.4f}")

    print(f"\n  ⚠️ Weak (0.10≤p<0.25): {len(weak_dims)}")
    for _, r in weak_dims.iterrows():
        print(f"    {r['dim']:<25} r={r['r']:+.3f}  p={r['p']:.4f}")

    print(f"\n  ❌ Noise (p≥0.25): {len(noise_dims)}")

    # Best combo
    if best_combos:
        bc = best_combos[0]
        print(f"\n  Best 3-dim model: {' × '.join(bc['combo'])}")
        print(f"    IS:  r={bc['r']:+.3f}  R²={bc['r2']:.4f}")
        print(f"    OOS: r={bc['oos_r']:+.3f}  R²={bc['oos_r2']:.4f}")


if __name__ == '__main__':
    main()
