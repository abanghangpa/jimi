"""
JIMI Framework — Core Backtest Engine
Orchestrates modules, ICS scoring, veto system, and trade lifecycle.
"""

import pandas as pd
import numpy as np
from src.config import CONFIG
from src.utils.data_handler import load_data, resample_ohlcv, fetch_btc_15m, fetch_recent
from src.utils.indicators import (
    calc_ema, calc_macd, calc_rsi, calc_atr, calc_vwap, calc_vol_ratio,
    calc_swing_bias, calc_phase0, calc_trend_state, compute_btc_correlation,
)
from src.modules.m1_macd import score_m1
from src.modules.m2_ema import score_m2
from src.modules.m3_vwap import score_m3
from src.modules.m4_cvd import calc_cvd_15m, detect_cvd_divergence_15m, calc_cvd_2h, detect_cvd_zero_cross, score_m4
from src.modules.m5_liquidation import score_m5, detect_cascade_setup
from src.modules.m6_derivatives import (
    fetch_all_derivatives, compute_oi_signals, compute_positioning_signals,
    score_derivatives, get_derivatives_summary, fetch_funding_rate,
)
from src.modules.m7_market_regime import m7_prepare_data, m7_get_row, score_m7
from src.modules.m8_funding import score_m8_funding
from src.modules.m9_volatility import RegimeState, compute_vol_regime, score_vol_regime
from src.modules.m10_macro import m10_prepare_data, m10_get_row, m10_compute_emas, score_m10_macro
from src.modules.m11_momentum import score_m11_mtf_momentum
from src.modules.m12_orderbook import score_m12_orderbook
from src.modules.m13_structure import score_m13
from src.modules.direction_resolver import resolve_direction
from src.modules.adaptive_direction import compute_adaptive_direction
from src.modules.veto_system import evaluate_vetoes, check_data_freshness
from src.modules.adaptive_weights import AdaptiveWeights
from src.modules.cross_asset import score_cross_asset
from src.modules.session import get_session


# ═══════════════════════════════════════════════════════════════
# TRADE CLASS
# ═══════════════════════════════════════════════════════════════

class Trade:
    def __init__(self, entry_time, direction, entry_price, sl, tp1, tp2, tp3,
                 size_pct, m1_dir, m2_status, m3_score, m4_status, m5_status, m5_score,
                 ics, phase0, reason, m1_score=0.5, m2_score=0.5, m4_score=0.5, m7_score=0.5,
                 m8_score=0.5, m8_status='SKIP', m9_score=0.5, m9_status='SKIP',
                 m10_score=0.5, m10_status='SKIP', m11_score=0.5, m11_status='SKIP',
                 m12_score=0.5, m12_status='SKIP', m13_score=0.5, m13_status='SKIP',
                 vol_regime='NEUTRAL',
                 trend_dir='NEUTRAL', trend_val=0.0, cross_asset_score=0.5,
                 session_name='UNKNOWN', veto_soft_penalty=0.0,
                 gatekeeper_passed=True, m7_details=None):
        self.entry_time = entry_time
        self.direction = direction
        self.entry_price = entry_price
        self.sl = sl
        self.tp1 = tp1
        self.tp2 = tp2
        self.tp3 = tp3
        self.size_pct = size_pct
        self.m1_dir = m1_dir
        self.m2_status = m2_status
        self.m3_score = m3_score
        self.m4_status = m4_status
        self.m5_status = m5_status
        self.m5_score = m5_score
        self.ics = ics
        self.phase0 = phase0
        self.reason = reason
        self.m1_score = m1_score
        self.m2_score = m2_score
        self.m4_score = m4_score
        self.m7_score = m7_score
        self.m8_score = m8_score
        self.m8_status = m8_status
        self.m9_score = m9_score
        self.m9_status = m9_status
        self.m10_score = m10_score
        self.m10_status = m10_status
        self.m11_score = m11_score
        self.m11_status = m11_status
        self.m12_score = m12_score
        self.m12_status = m12_status
        self.m13_score = m13_score
        self.m13_status = m13_status
        self.vol_regime = vol_regime
        self.trend_dir = trend_dir
        self.trend_val = trend_val
        self.cross_asset_score = cross_asset_score
        self.session_name = session_name
        self.veto_soft_penalty = veto_soft_penalty
        self.gatekeeper_passed = gatekeeper_passed
        self.m7_details = m7_details or {}
        # Lifecycle
        self.remaining = 1.0
        self.tp1_hit = False
        self.tp2_hit = False
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.pnl_pct = 0.0
        self.bars_held = 0

    def update_sl_trail(self):
        if self.tp2_hit:
            self.sl = self.tp1
        elif self.tp1_hit:
            self.sl = self.entry_price

    def close(self, price, time, reason, fraction=1.0):
        close_amount = min(fraction, self.remaining)
        pnl = ((price - self.entry_price) / self.entry_price if self.direction == 'LONG'
               else (self.entry_price - price) / self.entry_price)
        self.pnl_pct += pnl * close_amount
        self.remaining -= close_amount
        self.exit_price = price
        self.exit_time = time
        self.exit_reason = reason
        if self.remaining <= 0.001:
            self.remaining = 0

    @property
    def is_open(self):
        return self.remaining > 0.001


# ═══════════════════════════════════════════════════════════════
# ICS COMPOSITE SCORE
# ═══════════════════════════════════════════════════════════════

def calc_ics(m1_score, m2_score, m3_score, m4_score, m4_status, m5_score=0.5,
             m6_score=0.5, m7_score=0.5, m8_score=0.5, cross_asset_score=0.5,
             use_derivatives=False, use_m7=False, use_m8=False, use_cross_asset=False,
             cascade_dir='NONE', cascade_strength=0.0,
             m9_score=0.5, use_m9=False, m10_score=0.5, use_m10=False,
             m11_score=0.5, use_m11=False, m12_score=0.5, use_m12=False,
             m13_score=0.5, use_m13=False, config=None):
    cfg = config or CONFIG
    m4_contrib = m4_score if m4_status == 'PASS' else 0.5

    extra_modules = []
    if use_m7 and cfg.get('M7_ENABLED', False):
        extra_modules.append(('M7', m7_score, cfg['M7_WEIGHT']))
    if use_m8 and cfg.get('M8_ENABLED', False):
        extra_modules.append(('M8', m8_score, cfg.get('M8_WEIGHT', 0.10)))
    if use_cross_asset and cfg.get('CROSS_ASSET_ENABLED', False):
        extra_modules.append(('CA', cross_asset_score, cfg.get('CROSS_ASSET_BTC_WEIGHT', 0.08)))
    if use_m9 and cfg.get('M9_ENABLED', False):
        extra_modules.append(('M9', m9_score, cfg.get('M9_WEIGHT', 0.10)))
    if use_m10 and cfg.get('M10_ENABLED', False):
        extra_modules.append(('M10', m10_score, cfg.get('M10_WEIGHT', 0.10)))
    if use_m11 and cfg.get('M11_ENABLED', False):
        extra_modules.append(('M11', m11_score, cfg.get('M11_WEIGHT', 0.12)))
    if use_m12 and cfg.get('M12_ENABLED', False):
        extra_modules.append(('M12', m12_score, cfg.get('M12_WEIGHT', 0.05)))
    if use_m13 and cfg.get('M13_ENABLED', False):
        extra_modules.append(('M13', m13_score, cfg.get('M13_WEIGHT', 0.10)))

    if extra_modules:
        extra_w = sum(w for _, _, w in extra_modules)
        other_w = 1.0 - extra_w
        base_sum = (cfg['M1_WEIGHT'] + cfg['M2_WEIGHT'] +
                    cfg['M3_WEIGHT'] + cfg['M4_WEIGHT'] + cfg['M5_WEIGHT'])
        ics = (
            m1_score * (cfg['M1_WEIGHT'] / base_sum * other_w) +
            m2_score * (cfg['M2_WEIGHT'] / base_sum * other_w) +
            m3_score * (cfg['M3_WEIGHT'] / base_sum * other_w) +
            m4_contrib * (cfg['M4_WEIGHT'] / base_sum * other_w) +
            m5_score * (cfg['M5_WEIGHT'] / base_sum * other_w)
        )
        for _, score, weight in extra_modules:
            ics += score * weight
    elif use_derivatives:
        ics = (m1_score * cfg['M1_WEIGHT'] +
               m2_score * cfg['M2_WEIGHT'] +
               m3_score * cfg['M3_WEIGHT_DERIV'] +
               m4_contrib * cfg['M4_WEIGHT_DERIV'] +
               m5_score * cfg['M5_WEIGHT'] +
               m6_score * cfg['M6_WEIGHT'])
    else:
        ics = (m1_score * cfg['M1_WEIGHT'] +
               m2_score * cfg['M2_WEIGHT'] +
               m3_score * cfg['M3_WEIGHT'] +
               m4_contrib * cfg['M4_WEIGHT'] +
               m5_score * cfg['M5_WEIGHT'])

    if cascade_dir == 'WITH' and cascade_strength > 0:
        ics *= 1.0 + (cfg.get('CASCADE_MULTIPLIER', 1.12) - 1.0) * cascade_strength
    elif cascade_dir == 'AGAINST' and cascade_strength > 0:
        ics *= 1.0 - (1.0 - cfg.get('CASCADE_PENALTY', 0.85)) * cascade_strength

    effective_floor = cfg['ICS_FLOOR_M4_FALSE'] if m4_status == 'FAIL' else cfg['ICS_FLOOR']
    return ics, effective_floor


# ═══════════════════════════════════════════════════════════════
# GATEKEEPER
# ═══════════════════════════════════════════════════════════════

class GatekeeperResult:
    __slots__ = ('passed', 'blocked_by', 'size_mult', 'ics_boost', 'details')

    def __init__(self):
        self.passed = True
        self.blocked_by = []
        self.size_mult = 1.0
        self.ics_boost = 0.0
        self.details = {}

    def block(self, module, reason):
        self.passed = False
        self.blocked_by.append({'module': module, 'reason': reason})

    def summary(self):
        if self.passed:
            return f"PASS (mult={self.size_mult:.2f}, boost={self.ics_boost:+.3f})"
        return f"BLOCKED by {', '.join(b['module'] for b in self.blocked_by)}"


def run_gatekeepers(direction, vol_regime, m7_score, m7_status, m7_details,
                    m9_score, m9_status, m10_score, m10_status,
                    trend_dir, config=None):
    cfg = config or CONFIG
    result = GatekeeperResult()

    if cfg.get('M7_HARD_GATE', False) and m7_status != 'SKIP':
        gate_thresh = cfg.get('M7_GATE_THRESHOLD', 0.35)
        strong_thresh = cfg.get('M7_GATE_STRONG_THRESHOLD', 0.60)
        if direction == 'LONG' and m7_score < gate_thresh:
            result.block('M7', f'M7 {m7_score:.3f} < {gate_thresh}')
            return result
        elif direction == 'SHORT' and m7_score < gate_thresh:
            result.block('M7', f'M7 {m7_score:.3f} < {gate_thresh}')
            return result
        if (direction == 'LONG' and m7_score > strong_thresh) or \
           (direction == 'SHORT' and m7_score > strong_thresh):
            result.ics_boost += cfg.get('M7_GATE_STRONG_BOOST', 0.04)
            result.details['M7'] = 'strong_agree'

    if cfg.get('M10_ENABLED', False) and m10_status != 'SKIP':
        if direction == 'LONG' and m10_score < 0.25:
            result.block('M10', f'M10 {m10_score:.3f} strongly bearish')
            return result
        elif direction == 'SHORT' and m10_score < 0.25:
            result.block('M10', f'M10 {m10_score:.3f} strongly bullish')
            return result

    # M9 Regime Block — hard block on CRISIS and CHOP_HARD
    # M9 Regime Block — already handled in Phase 1 before direction resolution
    # (kept here as safety net for any edge case where regime changes mid-bar)
    if cfg.get('M9_ENABLED', False):
        block_regimes = cfg.get('M9_BLOCK_REGIMES', ['CRISIS'])
        if vol_regime in block_regimes:
            result.block('M9', f'regime={vol_regime}')
            return result

    if cfg.get('TREND_FILTER_ENABLED', False):
        if direction == 'LONG' and trend_dir == 'STRONG_DOWN':
            result.block('TREND', 'STRONG_DOWN vs LONG')
            return result
        elif direction == 'SHORT' and trend_dir == 'STRONG_UP':
            result.block('TREND', 'STRONG_UP vs SHORT')
            return result

    return result


# ═══════════════════════════════════════════════════════════════
# ENTRY FILTERS
# ═══════════════════════════════════════════════════════════════

def check_entry_filters(df_15m, idx, direction, swing_bias, phase0_val, atr_1h, config=None):
    cfg = config or CONFIG
    row = df_15m.iloc[idx]
    if not pd.isna(atr_1h) and atr_1h > 0:
        bar_move = abs(row['Close'] - row['Open'])
        if direction == 'LONG' and row['Close'] < row['Open']:
            if bar_move > cfg['BAR_MOVE_ATR'] * atr_1h:
                return False, "bar_move_against"
        if direction == 'SHORT' and row['Close'] > row['Open']:
            if bar_move > cfg['BAR_MOVE_ATR'] * atr_1h:
                return False, "bar_move_against"
    if not pd.isna(atr_1h) and row['Close'] > 0:
        if atr_1h / row['Close'] > cfg['ATR_FILTER_MAX']:
            return False, "atr_too_high"
    if phase0_val >= 0.90:
        return False, "phase0_red"
    return True, "ok"


def get_tp_multipliers(vol_ratio, config=None):
    cfg = config or CONFIG
    return cfg['TP2_ATR'], cfg['TP3_ATR']


# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

def run_backtest(csv_path, config=None, verbose=False, date_start=None, date_end=None):
    """Run the full backtest. Returns (trades, stats, df_15m)."""
    cfg = config or CONFIG

    print("=" * 70)
    print("  JIMI FRAMEWORK — Backtest Engine (M1-M12 + Adaptive Direction)")
    if date_start or date_end:
        print(f"  Date Range: {date_start or 'start'} → {date_end or 'end'}")
    print("=" * 70)

    print("\n[1/6] Loading data...")
    df_15m = load_data(csv_path)
    print(f"  15m bars loaded: {len(df_15m):,}")
    print(f"  Date range: {df_15m['Open time'].iloc[0]} → {df_15m['Open time'].iloc[-1]}")

    print("[2/6] Resampling to 1H, 2H, 4H, 1D...")
    df_1h = resample_ohlcv(df_15m, '1H')
    df_2h = resample_ohlcv(df_15m, '2H')
    df_4h = resample_ohlcv(df_15m, '4H')
    df_1d = resample_ohlcv(df_15m, '1D')
    print(f"  1H: {len(df_1h):,} | 2H: {len(df_2h):,} | 4H: {len(df_4h):,} | 1D: {len(df_1d):,}")

    print("[3/6] Computing indicators...")
    df_15m['vwap'] = calc_vwap(df_15m['High'], df_15m['Low'], df_15m['Close'], df_15m['Volume'], cfg['VWAP_LOOKBACK'])
    df_15m['vol_ma20'] = df_15m['Volume'].rolling(20).mean()
    taker_base = df_15m['Taker buy base asset volume']
    total_vol = df_15m['Volume']
    df_15m['taker_ratio'] = (taker_base / total_vol.replace(0, np.nan)).fillna(cfg['TAKER_FILLNA'])
    df_15m['atr'] = calc_atr(df_15m['High'], df_15m['Low'], df_15m['Close'], cfg['ATR_PERIOD'])
    df_15m['vol_ratio'] = calc_vol_ratio(df_15m['Volume'])

    df_1h['macd_line'], df_1h['macd_signal'], df_1h['macd_hist'] = calc_macd(
        df_1h['Close'], cfg['MACD_FAST'], cfg['MACD_SLOW'], cfg['MACD_SIGNAL'])
    df_1h['ema_fast'] = calc_ema(df_1h['Close'], cfg['EMA_FAST'])
    df_1h['ema_slow'] = calc_ema(df_1h['Close'], cfg['EMA_SLOW'])
    df_1h['atr'] = calc_atr(df_1h['High'], df_1h['Low'], df_1h['Close'], cfg['ATR_PERIOD'])

    df_4h['ema_fast'] = calc_ema(df_4h['Close'], cfg['EMA_FAST'])
    df_4h['ema_slow'] = calc_ema(df_4h['Close'], cfg['EMA_SLOW'])
    df_2h['ema_fast'] = calc_ema(df_2h['Close'], cfg['EMA_FAST'])
    df_2h['ema_slow'] = calc_ema(df_2h['Close'], cfg['EMA_SLOW'])

    df_15m['cvd_15m'] = calc_cvd_15m(df_15m)
    df_15m['cvd_divergence_15m'] = detect_cvd_divergence_15m(df_15m, cfg['CVD_LOOKBACK'], cfg['CVD_DIVERGENCE_WINDOW'])
    print(f"  CVD divergences (15m): {(df_15m['cvd_divergence_15m']=='BULLISH').sum()} bullish, {(df_15m['cvd_divergence_15m']=='BEARISH').sum()} bearish")

    df_2h['cvd_2h'] = calc_cvd_2h(df_2h)
    df_2h['cvd_zl_state'], df_2h['cvd_zl_cross_bar'], df_2h['cvd_zl_cross_dir'] = detect_cvd_zero_cross(df_2h)
    zl_up = (df_2h['cvd_zl_state'] == 'CROSS_UP').sum()
    zl_down = (df_2h['cvd_zl_state'] == 'CROSS_DOWN').sum()
    print(f"  CVD zero-line (2H):    {zl_up} cross-up, {zl_down} cross-down")

    df_1d['swing_bias'] = calc_swing_bias(df_1d)
    df_1d['phase0'] = calc_phase0(df_1d)
    df_1d['trend'], df_1d['trend_score'] = calc_trend_state(df_1d)

    if cfg.get('M1_RSI_ENABLED', False):
        df_1h['rsi'] = calc_rsi(df_1h['Close'], cfg.get('M1_RSI_PERIOD', 14))
        print(f"  RSI (1H) computed.")

    df_15m['rsi'] = calc_rsi(df_15m['Close'], 14)
    df_4h['macd_line'], df_4h['macd_signal'], df_4h['macd_hist'] = calc_macd(
        df_4h['Close'], cfg['MACD_FAST'], cfg['MACD_SLOW'], cfg['MACD_SIGNAL'])
    if 'rsi' not in df_1h.columns:
        df_1h['rsi'] = calc_rsi(df_1h['Close'], 14)
    print(f"  RSI (15m) + MACD (4H) computed for M11 divergence.")

    # M7: Market regime data
    m7_ethbtc_df, m7_btc_df = None, None
    if cfg.get('M7_ENABLED', False):
        print("[3b] Fetching M7 market regime data (ETH/BTC + BTC)...")
        m7_ethbtc_df, m7_btc_df = m7_prepare_data(df_15m)
        print(f"  M7 data: ETH/BTC={len(m7_ethbtc_df) if m7_ethbtc_df is not None else 0} days, BTC={len(m7_btc_df) if m7_btc_df is not None else 0} days")

    # M10: Cross-Asset Macro Data
    m10_data = None
    if cfg.get('M10_ENABLED', False):
        print("[3b-2] Fetching M10 cross-asset macro data...")
        try:
            m10_data = m10_prepare_data(df_15m)
            m10_data = m10_compute_emas(m10_data)
            for k, v in m10_data.items():
                print(f"    {k}: {len(v) if v is not None else 0} days")
        except Exception as e:
            print(f"    M10 fetch failed: {e}, macro scoring disabled")
            m10_data = None

    # Cross-Asset: BTC 15m
    btc_15m_df = None
    btc_corr_series = None
    if cfg.get('CROSS_ASSET_ENABLED', False):
        print("[3c] Fetching BTC/USDT 15m for cross-asset correlation...")
        try:
            btc_15m_df = fetch_btc_15m(df_15m['Open time'].iloc[0], df_15m['Open time'].iloc[-1])
            if btc_15m_df is not None and len(btc_15m_df) > 100:
                btc_corr_series = compute_btc_correlation(df_15m, btc_15m_df, cfg.get('CROSS_ASSET_LOOKBACK', 48))
                print(f"  BTC data: {len(btc_15m_df)} bars, correlation series computed")
            else:
                print(f"  BTC data: insufficient, cross-asset disabled")
                btc_15m_df = None
        except Exception as e:
            print(f"  BTC data: fetch failed ({e}), cross-asset disabled")
            btc_15m_df = None

    print("[4/7] Building timeframe index maps...")
    df_1h['_ts'] = df_1h['Open time'].values.astype('datetime64[ns]')
    df_2h['_ts'] = df_2h['Open time'].values.astype('datetime64[ns]')
    df_4h['_ts'] = df_4h['Open time'].values.astype('datetime64[ns]')
    df_1d['_ts'] = df_1d['Open time'].values.astype('datetime64[ns]')

    def find_tf_idx(ts, df_tf):
        idx = df_tf['_ts'].searchsorted(ts, side='right') - 1
        return max(idx, -1)

    warmup_time = df_1h['Open time'].iloc[min(cfg['WARMUP_BARS_1H'], len(df_1h)-1)]
    print(f"  Warmup: skip until {warmup_time}")

    # Post-crash cooldown windows
    _post_crash_windows = []
    if cfg.get('POST_CRASH_COOLDOWN', False):
        crash_thresh = cfg.get('POST_CRASH_THRESHOLD', 0.10)
        cooldown_bars = cfg.get('POST_CRASH_BARS', 192)
        for d_idx in range(1, len(df_1d)):
            prev_close = df_1d['Close'].iloc[d_idx - 1]
            curr_close = df_1d['Close'].iloc[d_idx]
            if prev_close > 0:
                day_chg = abs(curr_close - prev_close) / prev_close
                if day_chg > crash_thresh:
                    crash_ts = df_1d['Open time'].iloc[d_idx]
                    crash_end = crash_ts + pd.Timedelta(minutes=cooldown_bars * 15)
                    _post_crash_windows.append((crash_ts, crash_end))
        if _post_crash_windows:
            print(f"  Post-crash cooldown: {len(_post_crash_windows)} window(s)")

    print("[5/6] Running backtest...")

    # Cache funding rate
    cached_funding_rate = None
    if cfg.get('M8_ENABLED', False):
        try:
            fr_df = fetch_funding_rate("ETHUSDT", limit=1)
            if fr_df is not None and len(fr_df) > 0:
                cached_funding_rate = float(fr_df.iloc[-1].get('funding_rate', fr_df.iloc[-1].get('lastFundingRate', np.nan)))
                if np.isnan(cached_funding_rate):
                    cached_funding_rate = None
                else:
                    print(f"  Funding rate: {cached_funding_rate:.6f}")
        except Exception as e:
            print(f"  Funding rate fetch failed: {e}")

    trades, open_trades = [], []
    daily_trades, daily_pnl = {}, {}
    last_entry_bar = -999
    deriv_df = None
    regime_state = RegimeState()
    stats = {k: 0 for k in [
        'signals_checked', 'ics_blocked', 'filter_blocked', 'entries',
        'exits_sl', 'exits_tp1', 'exits_tp2', 'exits_tp3', 'exits_signal', 'exits_early',
        'm4_false_anchored', 'm5_pass', 'm5_fail', 'cascade_detected',
        'm1_neutral_skip', 'm3_fail', 'm2_neutral_long_skip', 'rolling_wr_skip',
        'dedup_skip', 'long_ics_skip', 'consec_pause',
        'ics_ceiling_skip', 'm4_required_skip',
        'bias_gate_skip', 'monthly_dd_skip', 'dir_veto_skip', 'trend_flip', 'trend_weak',
        'mtf_blocked', 'm8_pass', 'm8_fail',
        'm9_pass', 'm9_fail', 'm9_block',
        'm10_pass', 'm10_fail', 'm11_pass', 'm11_fail', 'm11_skip',
        'm12_pass', 'm12_fail', 'm12_skip',
        'm13_pass', 'm13_fail', 'm13_skip',
        'dir_resolved',
        'adaptive_dir_block', 'veto_hard_block', 'veto_hard_m9', 'veto_hard_data',
        'veto_hard_risk', 'veto_hard_dir', 'gate_block', 'gate_m7_block',
        'gate_m10_block', 'gate_trend_block', 'm2_veto_block', 'm5_hard_block',
        'session_asian_block', 'post_crash_block', 'veto_soft_applied', 'data_stale_block',
    ]}

    adaptive_tracker = None
    if cfg.get('ADAPTIVE_WEIGHTS_ENABLED', False):
        adaptive_tracker = AdaptiveWeights(
            decay=cfg.get('ADAPTIVE_DECAY', 0.95),
            min_mult=cfg.get('ADAPTIVE_MIN_MULT', 0.3),
            max_mult=cfg.get('ADAPTIVE_MAX_MULT', 2.0),
            warmup=cfg.get('ADAPTIVE_WARMUP_TRADES', 10),
        )

    for idx in range(len(df_15m)):
        row = df_15m.iloc[idx]
        ts = row['Open time']
        if ts < warmup_time:
            continue
        if date_start and str(ts) < date_start:
            continue
        if date_end and str(ts) > date_end:
            continue
        if pd.isna(row['taker_ratio']) or pd.isna(row['atr']):
            continue

        idx_1h = find_tf_idx(ts, df_1h)
        idx_2h = find_tf_idx(ts, df_2h)
        idx_4h = find_tf_idx(ts, df_4h)
        idx_1d = find_tf_idx(ts, df_1d)
        if idx_1h < 1 or idx_2h < 0 or idx_4h < 0 or idx_1d < 0:
            continue

        atr_1h = df_1h['atr'].iloc[idx_1h]
        swing_bias = df_1d['swing_bias'].iloc[idx_1d]
        phase0_val = df_1d['phase0'].iloc[idx_1d]
        trend_dir = df_1d['trend'].iloc[idx_1d]
        trend_val = df_1d['trend_score'].iloc[idx_1d]

        is_summer = ts.month in cfg.get('SUMMER_MONTHS', [6, 7, 8, 9])
        is_shoulder = ts.month in cfg.get('SHOULDER_MONTHS', [3, 10])

        # Check existing trades for SL/TP
        for trade in open_trades[:]:
            if not trade.is_open:
                continue
            high, low = row['High'], row['Low']
            trade.bars_held += 1

            early_exit_bars = cfg.get('EARLY_EXIT_BARS_SUMMER', cfg['EARLY_EXIT_BARS']) if is_summer else cfg['EARLY_EXIT_BARS']
            early_exit_loss = cfg.get('EARLY_EXIT_MIN_LOSS_SUMMER', cfg['EARLY_EXIT_MIN_LOSS']) if is_summer else cfg['EARLY_EXIT_MIN_LOSS']
            if trade.bars_held >= early_exit_bars and not trade.tp1_hit:
                current_pnl = ((row['Close'] - trade.entry_price) / trade.entry_price
                               if trade.direction == 'LONG'
                               else (trade.entry_price - row['Close']) / trade.entry_price)
                if current_pnl < -early_exit_loss:
                    trade.close(row['Close'], ts, 'EARLY_EXIT')
                    stats['exits_early'] += 1
                    continue

            if trade.direction == 'LONG' and low <= trade.sl:
                trade.close(trade.sl, ts, 'SL'); stats['exits_sl'] += 1; continue
            elif trade.direction == 'SHORT' and high >= trade.sl:
                trade.close(trade.sl, ts, 'SL'); stats['exits_sl'] += 1; continue

            if trade.tp1_hit and trade.tp2_hit:
                if (trade.direction == 'LONG' and high >= trade.tp3) or \
                   (trade.direction == 'SHORT' and low <= trade.tp3):
                    trade.close(trade.tp3, ts, 'TP3', trade.remaining); stats['exits_tp3'] += 1

            if trade.tp1_hit and not trade.tp2_hit:
                if trade.direction == 'LONG' and high >= trade.tp2:
                    frac = cfg['TP2_CLOSE'] / (1 - cfg['TP1_CLOSE'])
                    trade.close(trade.tp2, ts, 'TP2', frac); trade.tp2_hit = True; trade.update_sl_trail(); stats['exits_tp2'] += 1
                elif trade.direction == 'SHORT' and low <= trade.tp2:
                    frac = cfg['TP2_CLOSE'] / (1 - cfg['TP1_CLOSE'])
                    trade.close(trade.tp2, ts, 'TP2', frac); trade.tp2_hit = True; trade.update_sl_trail(); stats['exits_tp2'] += 1

            if not trade.tp1_hit:
                if is_summer:
                    tp1_close_frac = cfg.get('TP1_CLOSE_SUMMER', cfg['TP1_CLOSE'])
                elif is_shoulder:
                    tp1_close_frac = cfg.get('SHOULDER_TP1_CLOSE', cfg['TP1_CLOSE'])
                else:
                    tp1_close_frac = cfg['TP1_CLOSE']
                if trade.direction == 'LONG' and high >= trade.tp1:
                    trade.close(trade.tp1, ts, 'TP1', tp1_close_frac); trade.tp1_hit = True; trade.update_sl_trail(); stats['exits_tp1'] += 1
                elif trade.direction == 'SHORT' and low <= trade.tp1:
                    trade.close(trade.tp1, ts, 'TP1', tp1_close_frac); trade.tp1_hit = True; trade.update_sl_trail(); stats['exits_tp1'] += 1

        if adaptive_tracker is not None:
            for t in open_trades:
                if not t.is_open:
                    adaptive_tracker.update(t)

        open_trades = [t for t in open_trades if t.is_open]

        # Risk checks
        today = ts.date()
        if today not in daily_trades:
            daily_trades[today] = 0; daily_pnl[today] = 0.0

        today_closed = [t for t in trades if t.exit_time is not None and hasattr(t.exit_time, 'date') and t.exit_time.date() == today]
        daily_pnl[today] = sum(t.pnl_pct * t.size_pct for t in today_closed)
        max_trades_today = cfg.get('MAX_TRADES_DAY_SUMMER', cfg['MAX_TRADES_DAY']) if is_summer else (cfg.get('SHOULDER_MAX_TRADES_DAY', cfg['MAX_TRADES_DAY']) if is_shoulder else cfg['MAX_TRADES_DAY'])
        if daily_trades[today] >= max_trades_today:
            continue
        max_daily_loss = cfg.get('MAX_DAILY_LOSS_SUMMER', cfg['MAX_DAILY_LOSS']) if is_summer else cfg['MAX_DAILY_LOSS']
        if daily_pnl[today] <= -max_daily_loss:
            continue
        cooldown_bars = cfg.get('COOLDOWN_BARS_SUMMER', cfg['COOLDOWN_BARS']) if is_summer else (cfg.get('SHOULDER_COOLDOWN_BARS', cfg['COOLDOWN_BARS']) if is_shoulder else cfg['COOLDOWN_BARS'])
        if idx - last_entry_bar < cooldown_bars:
            continue
        phase0_block = cfg.get('PHASE0_SUMMER_BLOCK', 0.90) if is_summer else 0.90
        if phase0_val >= phase0_block:
            continue

        # Consecutive Loss Pause
        max_consec = cfg.get('MAX_CONSEC_LOSS_SUMMER', 999) if is_summer else cfg.get('MAX_CONSEC_LOSS', 999)
        if max_consec < 999 and len(trades) >= max_consec:
            recent = trades[-max_consec:]
            if all(t.pnl_pct < 0 for t in recent):
                last_exit = max(t.exit_time for t in recent if t.exit_time is not None)
                if last_exit is not None and hasattr(last_exit, 'total_seconds'):
                    pause_bars = cfg.get('CONSEC_LOSS_PAUSE_SUMMER', 8) if is_summer else cfg.get('CONSEC_LOSS_PAUSE_BARS', 8)
                    if (ts - last_exit).total_seconds() / 900 < pause_bars:
                        stats['consec_pause'] += 1
                        continue

        # Rolling Win Rate Circuit Breaker
        rolling_window = cfg.get('ROLLING_WR_WINDOW', 20)
        rolling_min = cfg.get('ROLLING_WR_MIN', 0.15)
        if len(trades) >= rolling_window:
            rolling_trades = trades[-rolling_window:]
            rolling_wr = sum(1 for t in rolling_trades if t.pnl_pct > 0) / rolling_window
            if rolling_wr < rolling_min:
                rolling_pnl = sum(t.pnl_pct * t.size_pct for t in rolling_trades)
                if rolling_pnl < 0:
                    stats['rolling_wr_skip'] += 1
                    continue

        # Post-Crash Cooldown
        if cfg.get('POST_CRASH_COOLDOWN', False):
            crash_block = False
            for crash_ts, crash_end_ts in _post_crash_windows:
                if crash_ts < ts < crash_end_ts:
                    crash_block = True
                    break
            if crash_block:
                stats['post_crash_block'] += 1
                continue

        stats['signals_checked'] += 1
        veto_soft_penalty = 0.0

        # ═══════════════════════════════════════════════════════════
        # PHASE 1: REGIME (M9) — What's the market climate?
        # ═══════════════════════════════════════════════════════════
        m9_score = 0.5; m9_status = 'SKIP'; vol_regime = 'NEUTRAL'; use_m9 = False
        m9_details = {}
        if cfg.get('M9_ENABLED', False):
            vol_regime, m9_raw, m9_vol_details = compute_vol_regime(
                df_15m, df_1h, idx, idx_1h, regime_state=regime_state)
            # Score with neutral direction first (direction not determined yet)
            m9_status, m9_score, m9_details = score_vol_regime(
                vol_regime, m9_raw, 'NEUTRAL', trend_dir)
            use_m9 = True

            # Hard block on CRISIS — skip immediately
            block_regimes = cfg.get('M9_BLOCK_REGIMES', ['CRISIS'])
            if vol_regime in block_regimes:
                stats['m9_block'] += 1
                continue

        # ═══════════════════════════════════════════════════════════
        # PHASE 2: DIRECTION (M13) — What's the structural bias?
        # ═══════════════════════════════════════════════════════════
        m13_score = 0.5; m13_status = 'SKIP'; m13_details = {}
        m13_bias = 'NEUTRAL'
        if cfg.get('M13_ENABLED', True):
            m13_status, m13_score, m13_details = score_m13(
                df_1h, idx_1h, 'NEUTRAL', df_15m, idx)
            m13_bias = m13_details.get('m13_bias', 'NEUTRAL')

        # M7 macro (needed for direction resolver)
        m7_score = 0.5; m7_status = 'SKIP'; m7_details = {}
        if cfg.get('M7_ENABLED', False) and m7_ethbtc_df is not None:
            eb_row, bt_row = m7_get_row(m7_ethbtc_df, m7_btc_df, ts)
            m7_status, m7_score, m7_details = score_m7(eb_row, bt_row, row.get('vol_ratio', np.nan), 'NEUTRAL')

        # ═══════════════════════════════════════════════════════════
        # PHASE 2c: RESOLVE DIRECTION — Climate + Structure + Macro
        # ═══════════════════════════════════════════════════════════
        direction, dir_size_mult, dir_details = resolve_direction(
            vol_regime, m9_score, m13_bias, m13_score, m13_details,
            m7_score=m7_score, m7_status=m7_status,
            swing_bias_1d=swing_bias, trend_dir=trend_dir, config=cfg,
        )

        if direction == 'NEUTRAL':
            stats['bias_gate_skip'] += 1
            continue

        # Re-score M9 and M7 with actual direction now that we know it
        if cfg.get('M9_ENABLED', False):
            m9_status, m9_score, m9_details = score_vol_regime(
                vol_regime, m9_raw, direction, trend_dir)
        if cfg.get('M7_ENABLED', False) and m7_ethbtc_df is not None:
            m7_status, m7_score, m7_details = score_m7(eb_row, bt_row, row.get('vol_ratio', np.nan), direction)
        if cfg.get('M13_ENABLED', True):
            m13_status, m13_score, m13_details = score_m13(
                df_1h, idx_1h, direction, df_15m, idx)

        # M1 + M2 still scored for ICS (not direction source)
        m1_dir, m1_score = score_m1(df_1h, idx_1h, cfg)
        m2_status, m2_score = score_m2(df_1h, df_2h, df_4h, df_1d, idx_1h, idx_2h, idx_4h, idx_1d)

        # M2 Veto (still applies)
        if cfg.get('M2_VETO_ENABLED', False):
            m2_veto_thresh = cfg.get('M2_VETO_THRESHOLD', 0.40)
            if direction == 'LONG' and m2_status == 'BEARISH' and m2_score < m2_veto_thresh:
                stats['m2_veto_block'] += 1
                continue
            if direction == 'SHORT' and m2_status == 'BULLISH' and m2_score < m2_veto_thresh:
                stats['m2_veto_block'] += 1
                continue

        # Adaptive Direction Bias
        if cfg.get('ADAPTIVE_DIR_ENABLED', False):
            ema_1h_f = df_1h['ema_fast'].iloc[idx_1h] if idx_1h >= 0 else None
            ema_1h_s = df_1h['ema_slow'].iloc[idx_1h] if idx_1h >= 0 else None
            ema_4h_f = df_4h['ema_fast'].iloc[idx_4h] if idx_4h >= 0 else None
            ema_4h_s = df_4h['ema_slow'].iloc[idx_4h] if idx_4h >= 0 else None
            ema_1d_f = calc_ema(df_1d['Close'], cfg['EMA_FAST']).iloc[idx_1d] if idx_1d >= 0 else None
            ema_1d_s = calc_ema(df_1d['Close'], cfg['EMA_SLOW']).iloc[idx_1d] if idx_1d >= 0 else None

            dir_bias, dir_allowed, dir_details = compute_adaptive_direction(
                trend_dir, trend_val,
                ema_1h_f, ema_1h_s, ema_4h_f, ema_4h_s, ema_1d_f, ema_1d_s,
                'NEUTRAL', recent_trades=trades[-8:] if trades else None,
                direction=direction, config=cfg,
            )
            if not dir_allowed:
                stats['adaptive_dir_block'] += 1
                continue

        # Legacy trend filter
        if cfg.get('TREND_FILTER_ENABLED', False):
            _trend_is_bull = trend_dir in ('STRONG_UP', 'UP')
            _trend_is_bear = trend_dir in ('STRONG_DOWN', 'DOWN')
            if cfg.get('TREND_BLOCK_COUNTER_TREND', False):
                if _trend_is_bear and direction == 'LONG':
                    stats['trend_flip'] += 1
                    continue
                if _trend_is_bull and direction == 'SHORT':
                    stats['trend_flip'] += 1
                    continue
            min_score = cfg.get('TREND_MIN_SCORE', 0.15)
            if abs(trend_val) < min_score:
                stats['trend_weak'] += 1
                continue

        m3_status, m3_score, m3_entry = score_m3(df_15m, idx, direction, cfg)
        if m3_status == 'FAIL':
            stats['m3_fail'] += 1
            continue

        if direction == 'LONG' and m2_status == 'NEUTRAL':
            stats['m2_neutral_long_skip'] += 1
            continue

        m4_status, m4_score, m4_div = score_m4(df_15m, df_2h, idx, idx_2h, direction, cfg)

        ics_pre, effective_floor = calc_ics(m1_score, m2_score, m3_score, m4_score, m4_status, 0.5, config=cfg)
        if m4_status == 'FAIL':
            stats['m4_false_anchored'] += 1
        threshold = cfg['ICS_THRESHOLD_CAUTION'] if phase0_val >= 0.40 else cfg['ICS_THRESHOLD_NORMAL']
        if is_summer:
            threshold += cfg.get('SUMMER_ICS_BOOST', 0)
        elif is_shoulder:
            threshold += cfg.get('SHOULDER_ICS_BOOST', 0)
        threshold += veto_soft_penalty
        if ics_pre < effective_floor or ics_pre < threshold:
            stats['ics_blocked'] += 1
            continue

        # M5 (lazy, cached every 4 bars)
        m5_cache_key = idx // 4
        if not hasattr(run_backtest, '_m5_cache') or run_backtest._m5_cache_key != m5_cache_key:
            m5_status, m5_score, m5_details = score_m5(df_15m, idx, direction, cfg,
                n_bins=cfg['M5_VP_BINS'], lookback=cfg['M5_VP_LOOKBACK'])
            cascade = detect_cascade_setup(df_15m, idx)
            run_backtest._m5_cache = (m5_status, m5_score, m5_details, cascade)
            run_backtest._m5_cache_key = m5_cache_key
        else:
            m5_status, m5_score, m5_details, cascade = run_backtest._m5_cache

        if m5_status == 'PASS':
            stats['m5_pass'] += 1
        else:
            stats['m5_fail'] += 1
        if cascade.get('cascade'):
            stats['cascade_detected'] += 1

        cascade_dir = m5_details.get('cascade_dir', 'NONE') if isinstance(m5_details, dict) else 'NONE'
        cascade_strength = m5_details.get('cascade_strength', 0.0) if isinstance(m5_details, dict) else 0.0

        # M7 already computed above (Phase 2)

        # M10
        if cfg.get('M8_ENABLED', False) and cached_funding_rate is not None:
            m8_status, m8_score, m8_details = score_m8_funding(cached_funding_rate, direction, cfg)
            use_m8 = True

        # M9 already computed above (Phase 1)
        # M7 already computed above (Phase 2)
        # M13 already computed above (Phase 2)
        m10_score = 0.5; m10_status = 'SKIP'; use_m10 = False
        if cfg.get('M10_ENABLED', False) and m10_data is not None:
            macro_row = m10_get_row(m10_data, ts)
            if macro_row:
                m10_status, m10_score, m10_details = score_m10_macro(macro_row, direction, trend_dir)
                use_m10 = True

        # M11
        m11_score = 0.5; m11_status = 'SKIP'; use_m11 = False
        if cfg.get('M11_ENABLED', False):
            m11_status, m11_score, m11_details = score_m11_mtf_momentum(
                df_15m, df_1h, df_4h, idx, idx_1h, idx_4h, direction)
            use_m11 = True

        # M12
        m12_score = 0.5; m12_status = 'SKIP'; use_m12 = False
        if cfg.get('M12_ENABLED', False) and not cfg.get('M12_LIVE_ONLY', True):
            m12_status, m12_score, m12_details = score_m12_orderbook(direction, live=False)
            use_m12 = True

        # Veto System
        if cfg.get('VETO_ENABLED', False):
            data_fresh = True; data_age = 0
            freshness_interval = cfg.get('DATA_FRESHNESS_CHECK_INTERVAL', 5)
            if cfg.get('DATA_FRESHNESS_ENABLED', False) and idx % freshness_interval == 0:
                if deriv_df is not None and len(deriv_df) > 0:
                    data_fresh, data_age, _ = check_data_freshness(
                        deriv_df, max_age_minutes=cfg.get('DATA_FRESHNESS_MAX_AGE_MIN', 20), current_time=ts)

            monthly_dd_hit = False
            monthly_dd_limit = cfg.get('MONTHLY_DD_CIRCUIT', 0)
            if monthly_dd_limit > 0:
                month_key = f"{ts.year}-{ts.month:02d}"
                month_trades_veto = [t for t in trades if t.exit_time is not None and hasattr(t.exit_time, 'year') and f"{t.exit_time.year}-{t.exit_time.month:02d}" == month_key]
                month_pnl_veto = sum(t.pnl_pct * t.size_pct for t in month_trades_veto)
                if month_pnl_veto <= -monthly_dd_limit:
                    monthly_dd_hit = True

            dir_conflict = False
            if cfg.get('DIR_VETO_ENABLED', False):
                m4_disagree = (direction == 'LONG' and m4_div == 'BEARISH') or (direction == 'SHORT' and m4_div == 'BULLISH')
                m5_disagree = (m5_status == 'FAIL')
                if m4_disagree and m5_disagree:
                    dir_conflict = True

            veto = evaluate_vetoes(
                cfg, vol_regime=vol_regime, data_fresh=data_fresh, data_age_minutes=data_age,
                monthly_dd_hit=monthly_dd_hit, dir_veto=dir_conflict,
                m9_status=m9_status, m10_status=m10_status, m11_status=m11_status,
            )

            if veto.hard_blocked:
                for v in veto.hard_vetoes:
                    key = f"veto_hard_{v['module'].lower()}"
                    stats[key] = stats.get(key, 0) + 1
                stats['veto_hard_block'] += 1
                continue

            veto_soft_penalty = veto.soft_penalty
            if veto.soft_vetoes:
                stats['veto_soft_applied'] += 1
        else:
            veto_soft_penalty = 0.0

        # Cross-Asset
        cross_asset_score = 0.5; use_cross_asset = False
        if cfg.get('CROSS_ASSET_ENABLED', False) and btc_15m_df is not None and btc_corr_series is not None:
            btc_corr_val = btc_corr_series.iloc[idx] if idx < len(btc_corr_series) else 0.5
            btc_change = 0.0
            if btc_15m_df is not None and len(btc_15m_df) > 4:
                btc_close_now = btc_15m_df['Close'].iloc[-1] if idx >= len(btc_15m_df) else btc_15m_df['Close'].iloc[min(idx, len(btc_15m_df)-1)]
                btc_close_1h_ago = btc_15m_df['Close'].iloc[max(0, min(idx-4, len(btc_15m_df)-1))]
                if btc_close_1h_ago > 0:
                    btc_change = (btc_close_now - btc_close_1h_ago) / btc_close_1h_ago
            cross_asset_score = score_cross_asset(row['Close'], btc_close_now if btc_15m_df is not None else None, btc_corr_val, btc_change, direction)
            use_cross_asset = True

        use_m7 = cfg.get('M7_ENABLED', False) and m7_ethbtc_df is not None

        # Gatekeeper
        gatekeeper = run_gatekeepers(
            direction, vol_regime, m7_score, m7_status, m7_details,
            m9_score, m9_status, m10_score, m10_status, trend_dir, config=cfg,
        )
        if not gatekeeper.passed:
            for b in gatekeeper.blocked_by:
                stats[f"gate_{b['module'].lower()}_block"] = stats.get(f"gate_{b['module'].lower()}_block", 0) + 1
            stats['gate_block'] += 1
            continue

        # ICS
        use_m13 = cfg.get('M13_ENABLED', False)
        ics, effective_floor = calc_ics(m1_score, m2_score, m3_score, m4_score, m4_status, m5_score,
                                        m7_score=m7_score, m8_score=m8_score, cross_asset_score=cross_asset_score,
                                        use_m7=use_m7, use_m8=use_m8, use_cross_asset=use_cross_asset,
                                        cascade_dir=cascade_dir, cascade_strength=cascade_strength,
                                        m9_score=m9_score, use_m9=use_m9, m10_score=m10_score, use_m10=use_m10,
                                        m11_score=m11_score, use_m11=use_m11, m12_score=m12_score, use_m12=use_m12,
                                        m13_score=m13_score, use_m13=use_m13, config=cfg)
        ics += gatekeeper.ics_boost

        # Session
        session_mult = 1.0; session_name = 'UNKNOWN'
        if cfg.get('SESSION_AWARENESS_ENABLED', False):
            session_name, session_mult = get_session(ts, cfg)
            if session_name == 'ASIAN' and cfg.get('SESSION_ASIAN_BLOCK', False):
                stats['session_asian_block'] += 1
                continue
            threshold *= (2.0 - session_mult)

        threshold += veto_soft_penalty

        # M5 Failure Penalty
        if m5_status == 'FAIL':
            threshold += cfg.get('M5_FAIL_ICS_BOOST', 0.06)
            if m5_score < cfg.get('M5_FAIL_HARD_THRESHOLD', 0.25):
                stats['m5_hard_block'] += 1
                continue

        if ics < effective_floor or ics < threshold:
            stats['ics_blocked'] += 1
            continue

        if ics > cfg.get('ICS_CEILING', 1.0):
            stats['ics_ceiling_skip'] += 1
            continue

        if m4_status == 'FAIL':
            stats['m4_required_skip'] += 1
            continue

        # Bias gate
        bad_gate_months = [3, 7, 9]
        if cfg.get('BIAS_GATE_ENABLED', False) and direction == 'LONG' and swing_bias == 'BEARISH' and ts.month in bad_gate_months:
            if ics < cfg.get('BIAS_GATE_LONG_ICS', 0.65):
                stats['bias_gate_skip'] += 1
                continue

        passed, reason = check_entry_filters(df_15m, idx, direction, swing_bias, phase0_val, atr_1h, config=cfg)
        if not passed:
            stats['filter_blocked'] += 1
            continue

        # Entry Dedup
        min_dist = cfg.get('MIN_ENTRY_DIST_PCT', 0)
        if min_dist > 0 and trades:
            last_trade = trades[-1]
            price_dist = abs(row['Close'] - last_trade.entry_price) / last_trade.entry_price
            if price_dist < min_dist:
                stats['dedup_skip'] += 1
                continue

        # Multi-TF Entry Confirmation
        if cfg.get('MTF_CONFIRM_ENABLED', False):
            mtf_block = False
            if cfg.get('MTF_1H_CANDLE_CHECK', False):
                if idx_1h >= 0:
                    h1_close = df_1h['Close'].iloc[idx_1h]
                    h1_open = df_1h['Open'].iloc[idx_1h]
                    h1_bullish = h1_close > h1_open
                    if direction == 'LONG' and not h1_bullish:
                        mtf_block = True
                    elif direction == 'SHORT' and h1_bullish:
                        mtf_block = True
            if cfg.get('MTF_4H_EMA_CHECK', False) and not mtf_block:
                if idx_4h >= 0:
                    ef4 = df_4h['ema_fast'].iloc[idx_4h]
                    es4 = df_4h['ema_slow'].iloc[idx_4h]
                    if direction == 'LONG' and ef4 < es4:
                        mtf_block = True
                    elif direction == 'SHORT' and ef4 > es4:
                        mtf_block = True
            if mtf_block:
                stats['mtf_blocked'] += 1
                continue

        # Position Sizing
        transition_range = cfg.get('TRANSITION_SCORE_RANGE', 0.20)
        is_transition = abs(trend_val) < transition_range

        size = cfg.get('SIZE_LONG', cfg['SIZE_STD']) if direction == 'LONG' else cfg['SIZE_STD']
        # Direction resolver regime-based size multiplier (primary sizing factor)
        size *= dir_size_mult
        if m2_status == 'NEUTRAL':
            size *= cfg['SIZE_M2_NEUTRAL']
        if phase0_val >= 0.40:
            size *= cfg['SIZE_CAUTION']
        if is_transition:
            size *= cfg.get('TRANSITION_SIZE_MULT', 0.50)
        if is_summer:
            size *= cfg.get('SUMMER_SIZE_MULT', 1.0)
        elif is_shoulder:
            size *= cfg.get('SHOULDER_SIZE_MULT', 1.0)
        if cfg.get('M7_ENABLED', False) and m7_ethbtc_df is not None:
            if m7_score < 0.35:
                size *= cfg.get('M7_SIZE_REDUCTION', 0.70)
            elif m7_score < 0.45:
                size *= cfg.get('M7_SIZE_MILD', 0.85)
        if cfg.get('M9_ENABLED', False):
            if vol_regime == 'CHOP_HARD':
                size *= cfg.get('M9_SIZE_CHOP_HARD', 0.0)  # blocked
            elif vol_regime == 'CHOP_MILD':
                size *= cfg.get('M9_SIZE_CHOP_MILD', 0.55)
            elif vol_regime == 'COMPRESSING':
                size *= cfg.get('M9_SIZE_COMPRESSING', 0.85)
        if adaptive_tracker is not None:
            size *= adaptive_tracker.size_multiplier(direction, m1_dir, m2_status, m3_score, m4_status, m5_status)
        if size < 0.01:
            continue

        # Entry
        entry_price = row['Close']
        atr_for_sl = atr_1h if not pd.isna(atr_1h) else row['atr']

        if is_transition:
            sl_std = cfg.get('SL_ATR_TRANSITION', 1.0)
            sl_hard_max = cfg.get('SL_HARD_MAX_PCT', 0.05)
        elif is_summer:
            sl_std = cfg.get('SL_ATR_STD_SUMMER', cfg['SL_ATR_STD'])
            sl_hard_max = cfg.get('SL_HARD_MAX_SUMMER', cfg['SL_HARD_MAX_PCT'])
        elif is_shoulder:
            sl_std = cfg.get('SHOULDER_SL_ATR', cfg['SL_ATR_STD'])
            sl_hard_max = cfg.get('SHOULDER_SL_HARD_MAX', cfg['SL_HARD_MAX_PCT'])
        else:
            sl_std = cfg['SL_ATR_STD']
            sl_hard_max = cfg['SL_HARD_MAX_PCT']

        sl_dist = min(sl_std * atr_for_sl, sl_hard_max * entry_price)

        if is_transition:
            tp1_atr = cfg.get('TP1_ATR_TRANSITION', 1.2)
        elif is_summer:
            tp1_atr = cfg.get('TP1_ATR_SUMMER', cfg['TP1_ATR'])
        elif is_shoulder:
            tp1_atr = cfg.get('SHOULDER_TP1_ATR', cfg['TP1_ATR'])
        else:
            tp1_atr = cfg['TP1_ATR']

        tp1_dist = tp1_atr * atr_for_sl
        tp2_mult, tp3_mult = get_tp_multipliers(row.get('vol_ratio', np.nan), config=cfg)
        tp2_dist, tp3_dist = tp2_mult * atr_for_sl, tp3_mult * atr_for_sl

        if direction == 'LONG':
            sl, tp1, tp2, tp3 = entry_price - sl_dist, entry_price + tp1_dist, entry_price + tp2_dist, entry_price + tp3_dist
        else:
            sl, tp1, tp2, tp3 = entry_price + sl_dist, entry_price - tp1_dist, entry_price - tp2_dist, entry_price - tp3_dist

        trade = Trade(ts, direction, entry_price, sl, tp1, tp2, tp3, size,
                      m1_dir, m2_status, m3_score, m4_status, m5_status, m5_score,
                      ics, phase0_val,
                      f"M9={vol_regime} M13={m13_bias} M1={m1_dir} M2={m2_status} M3={m3_status} M4={m4_status} M5={m5_status} ICS={ics:.3f}",
                      m1_score=m1_score, m2_score=m2_score, m4_score=m4_score, m7_score=m7_score,
                      m8_score=m8_score, m8_status=m8_status, m9_score=m9_score, m9_status=m9_status,
                      m10_score=m10_score, m10_status=m10_status, m11_score=m11_score, m11_status=m11_status,
                      m12_score=m12_score, m12_status=m12_status,
                      m13_score=m13_score, m13_status=m13_status,
                      vol_regime=vol_regime, trend_dir=trend_dir, trend_val=trend_val,
                      cross_asset_score=cross_asset_score, session_name=session_name,
                      veto_soft_penalty=veto_soft_penalty, gatekeeper_passed=gatekeeper.passed,
                      m7_details=m7_details)
        open_trades.append(trade); trades.append(trade)
        daily_trades[today] += 1; stats['entries'] += 1
        last_entry_bar = idx

        if verbose and stats['entries'] <= 50:
            print(f"  ENTRY #{stats['entries']}: {ts} {direction} @ {entry_price:.2f} "
                  f"SL={sl:.2f} TP1={tp1:.2f} ICS={ics:.3f} M5={m5_status}({m5_score:.2f}) M7={m7_score:.2f} size={size:.2f}")

    # Close remaining
    if open_trades:
        last_row = df_15m.iloc[-1]
        for trade in open_trades:
            if trade.is_open:
                trade.close(last_row['Close'], last_row['Open time'], 'END'); stats['exits_signal'] += 1
        if adaptive_tracker is not None:
            for t in open_trades:
                if not t.is_open:
                    adaptive_tracker.update(t)

    print("\n[7/7] Computing results...")
    if adaptive_tracker is not None:
        print(adaptive_tracker.summary())
    return trades, stats, df_15m
