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
from src.modules.m1_macd_v2 import score_m1_v2 as score_m1
from src.modules.m2_ema import score_m2
from src.modules.m3_vwap import score_m3
from src.modules.m4_cvd import calc_cvd_15m, detect_cvd_divergence_15m, calc_cvd_2h, detect_cvd_zero_cross, score_m4
from src.modules.intrabar_cvd import compute_intrabar_delta, aggregate_to_timeframe, detect_intrabar_divergence, score_intrabar_divergence
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
from src.modules.direction_resolver import resolve_direction, score_targets
from src.modules.adaptive_direction import compute_adaptive_direction
from src.modules.veto_system import evaluate_vetoes, check_data_freshness
from src.modules.adaptive_weights import AdaptiveWeights
from src.modules.cross_asset import score_cross_asset
from src.modules.session import get_session
from src.modules.coherence_liquidity import check_coherence, compute_liquidity_aware_tp, compute_stop_risk
from src.sl_tp import calc_trade_levels, check_sweep_gate
from src.modules.m14_sweep import score_m14
from src.modules.m17_resistance_quality import score_resistance_quality
from src.modules.entry_optimizer import detect_wick_reclaim
from src.modules.m18_squeeze import detect_squeeze_v6 as detect_squeeze, SQUEEZE_V6_DEFAULTS as SQUEEZE_V5_DEFAULTS
from src.modules.m19_breakout_confirm import check_breakout_filters
from src.modules.m20_failed_breakout import score_m20


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
                 m14_score=0.5, m14_status='SKIP',
                 vol_regime='NEUTRAL',
                 trend_dir='NEUTRAL', trend_val=0.0, cross_asset_score=0.5,
                 session_name='UNKNOWN', veto_soft_penalty=0.0,
                 gatekeeper_passed=True, m7_details=None,
                 tp1_close_frac=None, tp2_close_frac=None,
                 squeeze_type='NONE', squeeze_score=0.0, squeeze_strong=False,
                 squeeze_trigger_type='NONE', squeeze_failed_breakout=False,
                 squeeze_box_type='UNKNOWN', squeeze_lifecycle='NONE'):
        self.entry_time = entry_time
        self.direction = direction
        self.entry_price = entry_price
        self.sl = sl
        self.tp1 = tp1
        self.tp2 = tp2
        self.tp3 = tp3
        self.tp1_close_frac = tp1_close_frac  # liquidity-adjusted
        self.tp2_close_frac = tp2_close_frac  # liquidity-adjusted
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
        self.m14_score = m14_score
        self.m14_status = m14_status
        self.vol_regime = vol_regime
        self.trend_dir = trend_dir
        self.trend_val = trend_val
        self.cross_asset_score = cross_asset_score
        self.session_name = session_name
        self.veto_soft_penalty = veto_soft_penalty
        self.gatekeeper_passed = gatekeeper_passed
        self.m7_details = m7_details or {}
        # Squeeze tracking
        self.squeeze_type = squeeze_type
        self.squeeze_score = squeeze_score
        self.squeeze_strong = squeeze_strong
        self.squeeze_trigger_type = squeeze_trigger_type
        self.squeeze_failed_breakout = squeeze_failed_breakout
        self.squeeze_box_type = squeeze_box_type
        self.squeeze_lifecycle = squeeze_lifecycle
        # Lifecycle
        self.remaining = 1.0
        self.tp1_hit = False
        self.tp2_hit = False
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.pnl_pct = 0.0
        self.bars_held = 0
        # Regime lifecycle tracking
        self.entry_regime = vol_regime
        self.exit_regime = None
        self.regime_history = [vol_regime]  # regime at each bar during trade
        self.regime_transitions = 0  # how many times regime changed during trade
        self.dir_size_mult = 1.0  # will be set at entry

    def update_sl_trail(self):
        if self.tp2_hit:
            self.sl = self.tp1
        elif self.tp1_hit:
            self.sl = self.entry_price

    def update_regime(self, current_regime):
        """Track regime changes during the life of this trade."""
        if self.regime_history and self.regime_history[-1] != current_regime:
            self.regime_transitions += 1
        self.regime_history.append(current_regime)

    def close(self, price, time, reason, fraction=1.0, exit_regime=None):
        close_amount = min(fraction, self.remaining)
        pnl = ((price - self.entry_price) / self.entry_price if self.direction == 'LONG'
               else (self.entry_price - price) / self.entry_price)
        self.pnl_pct += pnl * close_amount
        self.remaining -= close_amount
        self.exit_price = price
        self.exit_time = time
        self.exit_reason = reason
        if exit_regime is not None:
            self.exit_regime = exit_regime
        elif self.regime_history:
            self.exit_regime = self.regime_history[-1]
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
             m13_score=0.5, use_m13=False, m14_score=0.5, use_m14=False,
             m17_score=0.5, use_m17=False,
             m20_score=0.5, use_m20=False,
             config=None):
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
    if use_m14 and cfg.get('M14_ENABLED', False):
        extra_modules.append(('M14', m14_score, cfg.get('M14_WEIGHT', 0.08)))
    if use_m17 and cfg.get('M17_ENABLED', False):
        extra_modules.append(('M17', m17_score, cfg.get('M17_WEIGHT', 0.05)))
    if use_m20 and cfg.get('M20_ENABLED', False):
        extra_modules.append(('M20', m20_score, cfg.get('M20_WEIGHT', 0.10)))

    base_sum = (cfg['M1_WEIGHT'] + cfg['M2_WEIGHT'] +
                cfg['M3_WEIGHT'] + cfg['M4_WEIGHT'] + cfg['M5_WEIGHT'])

    if extra_modules:
        extra_w = sum(w for _, _, w in extra_modules)
        # Normalize base weights to fill remaining space after extras
        # so total always sums to 1.0
        base_w = max(1.0 - extra_w, 0.0)
        ics = (
            m1_score * (cfg['M1_WEIGHT'] / base_sum * base_w) +
            m2_score * (cfg['M2_WEIGHT'] / base_sum * base_w) +
            m3_score * (cfg['M3_WEIGHT'] / base_sum * base_w) +
            m4_contrib * (cfg['M4_WEIGHT'] / base_sum * base_w) +
            m5_score * (cfg['M5_WEIGHT'] / base_sum * base_w)
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
        # Normalize base weights to sum to 1.0
        ics = (
            m1_score * (cfg['M1_WEIGHT'] / base_sum) +
            m2_score * (cfg['M2_WEIGHT'] / base_sum) +
            m3_score * (cfg['M3_WEIGHT'] / base_sum) +
            m4_contrib * (cfg['M4_WEIGHT'] / base_sum) +
            m5_score * (cfg['M5_WEIGHT'] / base_sum)
        )

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
    # Phase0 minimum block — DISABLED (scanner reports phase0 in
    # invalidation info only; does not gate signals on it)
    # phase0_min = cfg.get('PHASE0_MIN_BLOCK', 0.0)
    # if phase0_min > 0 and phase0_val < phase0_min:
    #     return False, "phase0_death_zone"

    # ── Survival filter: momentum gate ──
    # Reject entries when recent bars show no directional momentum.
    # Uses 3 signals: recent close direction, vol expansion, and bar consistency.
    survival_enabled = cfg.get('SURVIVAL_FILTER_ENABLED', True)
    if survival_enabled and idx >= 12:
        lookback = cfg.get('SURVIVAL_LOOKBACK', 12)  # 3 hours on 15m
        closes = df_15m['Close'].iloc[idx - lookback:idx + 1].values.astype(float)
        opens = df_15m['Open'].iloc[idx - lookback:idx + 1].values.astype(float)
        highs = df_15m['High'].iloc[idx - lookback:idx + 1].values.astype(float)
        lows = df_15m['Low'].iloc[idx - lookback:idx + 1].values.astype(float)

        # 1. Directional move: net change over lookback
        net_move = (closes[-1] - closes[0]) / closes[0]
        if direction == 'LONG' and net_move < cfg.get('SURVIVAL_MIN_MOVE', -0.002):
            return False, "survival_no_momentum"
        if direction == 'SHORT' and net_move > cfg.get('SURVIVAL_MIN_MOVE', -0.002) * -1:
            return False, "survival_no_momentum"

        # 2. Bar consistency: how many bars closed in trade direction
        bull_bars = sum(1 for i in range(len(closes)) if closes[i] >= opens[i])
        bear_bars = len(closes) - bull_bars
        if direction == 'LONG':
            consistency = bull_bars / len(closes)
        else:
            consistency = bear_bars / len(closes)
        if consistency < cfg.get('SURVIVAL_MIN_CONSISTENCY', 0.35):
            return False, "survival_inconsistent"

        # 3. Vol expansion: recent ATR vs longer ATR
        if not pd.isna(atr_1h) and atr_1h > 0:
            recent_range = np.mean(highs[-4:] - lows[-4:])  # last 1h
            if recent_range / atr_1h < cfg.get('SURVIVAL_VOL_RATIO', 0.30):
                return False, "survival_low_vol"

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

    # M4b: Intrabar CVD (LucF-style, adapted for backtest with 15m bars)
    # Uses close vs open of each 15m bar to classify volume direction
    if cfg.get('M4B_INTRABAR_ENABLED', True):
        intrabar_delta = compute_intrabar_delta(df_15m)
        df_15m['cvd_intrabar'] = intrabar_delta.cumsum()
        df_15m['cvd_intrabar_div'] = detect_intrabar_divergence(
            df_15m.rename(columns={'cvd_intrabar': 'cvd'}), lookback=24, window=12)
        bull = (df_15m['cvd_intrabar_div'] == 'BULLISH').sum()
        bear = (df_15m['cvd_intrabar_div'] == 'BEARISH').sum()
        print(f"  CVD intrabar (M4b):    {bull} bullish, {bear} bearish divergences")
    else:
        df_15m['cvd_intrabar'] = 0.0
        df_15m['cvd_intrabar_div'] = 'NONE'

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

    # Funding rate — cached once at start (API returns current rate, not historical)
    # TODO: fetch historical funding rates from Binance for accurate backtest
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
    regime_state = RegimeState(config=cfg)
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
        'coherence_block', 'coherence_penalty',
        'm14_pass', 'm14_fail', 'm14_skip',
        'wick_reclaim_bonus', 'wick_reclaim_penalty', 'wick_slice_block',
        'exits_time_stop',
        # Squeeze stats
        'squeeze_detected', 'squeeze_triggered', 'squeeze_pending',
        'squeeze_failed_breakout', 'squeeze_entries', 'squeeze_wins',
        'squeeze_long', 'squeeze_short', 'squeeze_direction_block',
        'squeeze_confirmed', 'squeeze_not_confirmed',
        'breakout_confirmed', 'breakout_weak', 'breakout_rejected',
        'm17_pass', 'm17_skip',
        'm20_pass', 'm20_fail', 'm20_skip', 'm20_direct_signal',
        'm20_override', 'm20_ics_bypass',
    ]}

    adaptive_tracker = None
    if cfg.get('ADAPTIVE_WEIGHTS_ENABLED', False):
        adaptive_tracker = AdaptiveWeights(
            decay=cfg.get('ADAPTIVE_DECAY', 0.95),
            min_mult=cfg.get('ADAPTIVE_MIN_MULT', 0.3),
            max_mult=cfg.get('ADAPTIVE_MAX_MULT', 2.0),
            warmup=cfg.get('ADAPTIVE_WARMUP_TRADES', 10),
        )

    # Squeeze tracking state
    squeeze_enabled = cfg.get('SQUEEZE_ENABLED', True)
    squeeze_compression_history = []  # list of (range48, vol_ratio, bar_range, taker_ratio)
    last_squeeze_bar = -999
    squeeze_last_squeeze_bar = -999  # squeeze-specific cooldown

    for idx in range(len(df_15m)):
        row = df_15m.iloc[idx]
        ts = row['Open time']
        # Squeeze defaults per bar (actual detection happens later)
        squeeze_result = {'squeeze_type': 'NONE', 'squeeze_status': 'NONE', 'squeeze_score': 0.0, 'ics_boost': 0.0, 'direction': 'NEUTRAL'}
        squeeze_confirmed = False
        if ts < warmup_time:
            continue
        if date_start and str(ts) < date_start:
            continue
        if date_end and str(ts) > date_end:
            continue
        if pd.isna(row['taker_ratio']) or pd.isna(row['atr']):
            continue

        # Build compression history for squeeze detector (rolling 48-bar window)
        if squeeze_enabled and idx >= 47:
            _r48_h = float(df_15m['High'].iloc[max(0, idx-47):idx+1].max())
            _r48_l = float(df_15m['Low'].iloc[max(0, idx-47):idx+1].min())
            _close_val = float(row['Close'])
            _r48_pct = (_r48_h - _r48_l) / _close_val * 100 if _close_val > 0 else 5.0
            _vr = float(row['Volume'] / row['vol_ma20']) if row['vol_ma20'] > 0 else 1.0
            _br = (float(row['High']) - float(row['Low'])) / _close_val * 100 if _close_val > 0 else 0.5
            _tr = float(row['taker_ratio'])
            squeeze_compression_history.append((_r48_pct, _vr, _br, _tr))
            # Keep only last 96 bars (24h) to avoid memory growth
            if len(squeeze_compression_history) > 96:
                squeeze_compression_history = squeeze_compression_history[-96:]

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

            # ── Time-stop: exit if TP1 not hit within N bars ──
            # ADAPTIVE: scales with TP1 distance. Wider TP targets need
            # more time to develop. Formula:
            #   bars = max(BASE, BASE + tp1_pct * PER_PCT), capped at MAX
            # Examples (BASE=20, PER_PCT=25, MAX=60):
            #   TP1 0.2% away → 20 bars (minimum)
            #   TP1 0.5% away → 32 bars
            #   TP1 1.0% away → 45 bars
            #   TP1 1.5% away → 57 bars
            ts_base = cfg.get('TIME_STOP_BASE_BARS', cfg.get('TIME_STOP_BARS', 30))
            ts_per_pct = cfg.get('TIME_STOP_PER_PCT', 25)  # extra bars per 1% TP1 distance
            ts_max = cfg.get('TIME_STOP_MAX_BARS', 60)
            if ts_base > 0 and not trade.tp1_hit:
                tp1_pct = abs(trade.tp1 - trade.entry_price) / trade.entry_price * 100
                time_stop_bars = min(ts_max, max(ts_base, int(ts_base + tp1_pct * ts_per_pct)))
            else:
                time_stop_bars = ts_base
            if time_stop_bars > 0 and trade.bars_held >= time_stop_bars and not trade.tp1_hit:
                trade.close(row['Close'], ts, 'TIME_STOP')
                stats['exits_time_stop'] = stats.get('exits_time_stop', 0) + 1
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
                    _tp2_frac = trade.tp2_close_frac if trade.tp2_close_frac is not None else cfg['TP2_CLOSE']
                    _tp1_frac = trade.tp1_close_frac if trade.tp1_close_frac is not None else cfg['TP1_CLOSE']
                    frac = _tp2_frac / (1 - _tp1_frac)
                    trade.close(trade.tp2, ts, 'TP2', frac); trade.tp2_hit = True; trade.update_sl_trail(); stats['exits_tp2'] += 1
                elif trade.direction == 'SHORT' and low <= trade.tp2:
                    _tp2_frac = trade.tp2_close_frac if trade.tp2_close_frac is not None else cfg['TP2_CLOSE']
                    _tp1_frac = trade.tp1_close_frac if trade.tp1_close_frac is not None else cfg['TP1_CLOSE']
                    frac = _tp2_frac / (1 - _tp1_frac)
                    trade.close(trade.tp2, ts, 'TP2', frac); trade.tp2_hit = True; trade.update_sl_trail(); stats['exits_tp2'] += 1

            if not trade.tp1_hit:
                _trade_in_chop = trade.entry_regime in ('CHOP_MILD', 'CHOP_MILD_BEAR', 'CHOP_MILD_BULL')
                if _trade_in_chop:
                    tp1_close_frac = 1.0  # exit fully at TP1 in chop
                elif is_summer:
                    tp1_close_frac = cfg.get('TP1_CLOSE_SUMMER', cfg['TP1_CLOSE'])
                elif is_shoulder:
                    tp1_close_frac = cfg.get('SHOULDER_TP1_CLOSE', cfg['TP1_CLOSE'])
                else:
                    tp1_close_frac = trade.tp1_close_frac if trade.tp1_close_frac is not None else cfg['TP1_CLOSE']
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
        # Phase0 minimum block — DISABLED (scanner doesn't gate on phase0;
        # it reports phase0 in invalidation info only, so engine must match)
        # phase0_min = cfg.get('PHASE0_MIN_BLOCK', 0.0)
        # if phase0_min > 0 and phase0_val < phase0_min:
        #     stats['bias_gate_skip'] += 1
        #     continue

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

        # Reset per-iteration module state (prevent stale values from prior bar)
        m8_score = 0.5; m8_status = 'SKIP'; use_m8 = False

        # ═══════════════════════════════════════════════════════════
        # PHASE 1: REGIME (M9) — What's the market climate?
        # ═══════════════════════════════════════════════════════════
        m9_score = 0.5; m9_status = 'SKIP'; vol_regime = 'NEUTRAL'; use_m9 = False
        m9_details = {}
        if cfg.get('M9_ENABLED', False):
            vol_regime, m9_raw, m9_vol_details = compute_vol_regime(
                df_15m, df_1h, idx, idx_1h, regime_state=regime_state, config=cfg)
            # Score with neutral direction first (direction not determined yet)
            m9_status, m9_score, m9_details = score_vol_regime(
                vol_regime, m9_raw, 'NEUTRAL', trend_dir)
            use_m9 = True

            # Hard block on CRISIS — skip immediately
            block_regimes = cfg.get('M9_BLOCK_REGIMES', ['CRISIS'])
            if vol_regime in block_regimes:
                stats['m9_block'] += 1
                continue

        # Update regime tracking for open trades (after M9, before entries)
        for trade in open_trades:
            if trade.is_open:
                trade.update_regime(vol_regime)

        # ═══════════════════════════════════════════════════════════
        # PHASE 2: DIRECTION (M13) — What's the structural bias?
        # ═══════════════════════════════════════════════════════════
        # M13 defers to M9 during chop regimes — diagnostic data shows
        # M13 is anti-predictive when it agrees with M9 during chop.
        # The CHOP_MILD_BEAR/BULL directional split is the primary
        # direction source during chop; M13 only adds value when M9 is NEUTRAL.
        m13_score = 0.5; m13_status = 'SKIP'; m13_details = {}
        m13_bias = 'NEUTRAL'
        _chop_regimes = ('CHOP_MILD', 'CHOP_MILD_BEAR', 'CHOP_MILD_BULL', 'CHOP_HARD')
        _m13_defer_in_chop = cfg.get('M13_DEFER_IN_CHOP', True)
        _in_chop = _m13_defer_in_chop and vol_regime in _chop_regimes
        if cfg.get('M13_ENABLED', True):
            # Always compute structural data (swing levels, FVGs, OBs)
            # so M14 and wick reclaim have access even during chop.
            # Directional bias and ICS score are suppressed during chop
            # (M13 is anti-predictive when it agrees with M9 during chop).
            _m13_status, _m13_score_raw, m13_details = score_m13(
                df_1h, idx_1h, 'NEUTRAL', df_15m, idx)
            if not _in_chop:
                m13_status = _m13_status
                m13_score = _m13_score_raw
                m13_bias = m13_details.get('m13_bias', 'NEUTRAL')
            # else: m13_score stays 0.5, m13_bias stays NEUTRAL
            # but m13_details has swing levels for M14/wick reclaim

        # M7 macro (needed for direction resolver)
        m7_score = 0.5; m7_status = 'SKIP'; m7_details = {}
        if cfg.get('M7_ENABLED', False) and m7_ethbtc_df is not None:
            eb_row, bt_row = m7_get_row(m7_ethbtc_df, m7_btc_df, ts)
            m7_status, m7_score, m7_details = score_m7(eb_row, bt_row, row.get('vol_ratio', np.nan), 'NEUTRAL')

        # ═══════════════════════════════════════════════════════════
        # PHASE 2c: RESOLVE DIRECTION — Climate + Structure + Macro + Targets
        # ═══════════════════════════════════════════════════════════
        # Pre-compute volume profile for target scoring
        _tgt_cache_key = idx // 4
        if not hasattr(run_backtest, '_tgt_cache') or run_backtest._tgt_cache_key != _tgt_cache_key:
            _highs = df_15m['High'].values.astype(float)
            _lows = df_15m['Low'].values.astype(float)
            _closes = df_15m['Close'].values.astype(float)
            _volumes = df_15m['Volume'].values.astype(float)
            from src.modules.m5_liquidation import build_volume_profile, find_magnets, find_gaps, find_support_resistance
            _tc, _tp, _te = build_volume_profile(
                _highs[:idx+1], _lows[:idx+1], _closes[:idx+1], _volumes[:idx+1],
                n_bins=cfg.get('M5_VP_BINS', 50), lookback=cfg.get('M5_VP_LOOKBACK', 672))
            _magnets = find_magnets(_tc, _tp) if _tc is not None else []
            _gaps = find_gaps(_tc, _tp) if _tc is not None else []
            _sr = find_support_resistance(df_15m, idx)
            run_backtest._tgt_cache = (_magnets, _gaps, _sr)
            run_backtest._tgt_cache_key = _tgt_cache_key
        else:
            _magnets, _gaps, _sr = run_backtest._tgt_cache

        _atr_1h = df_1h['atr'].iloc[idx_1h] if idx_1h >= 0 else None
        _price = float(row['Close'])
        _long_tgt, _long_det = score_targets(_price, _magnets, _gaps, _sr, 'LONG', atr_1h=_atr_1h)
        _short_tgt, _short_det = score_targets(_price, _magnets, _gaps, _sr, 'SHORT', atr_1h=_atr_1h)

        # ── M20 Pre-compute (before direction resolver) ──
        # M20 detects failed breakouts independently. Run it early so its
        # contrarian direction can feed into the direction resolver as an
        # override when a strong failed breakout is detected.
        _m20_pre_score = None
        _m20_pre_dir = None
        if cfg.get('M20_ENABLED', True):
            try:
                _sr_for_m20 = _sr if _sr else None
                _mag_for_m20 = _magnets if _magnets else None
                _m20_pre_status, _m20_pre_score, _m20_pre_result = score_m20(
                    df_15m, idx, 'NEUTRAL',
                    sr_levels=_sr_for_m20, magnets=_mag_for_m20,
                    config=cfg, atr_1h=_atr_1h)
                if _m20_pre_result and _m20_pre_result.get('status') == 'FAILED':
                    _m20_pre_dir = _m20_pre_result.get('contrarian_direction')
            except Exception:
                pass

        # Compute nearest_liq_direction from unswept magnets (aligned with scanner)
        _nearest_liq_dir = None
        if _magnets:
            _above = [(p, s) for p, v, s in _magnets if p > _price]
            _below = [(p, s) for p, v, s in _magnets if p < _price]
            if _above and _below:
                _nearest_above = min(_above, key=lambda x: x[0] - _price)
                _nearest_below = min(_below, key=lambda x: _price - x[0])
                _above_dist = _nearest_above[0] - _price
                _below_dist = _price - _nearest_below[0]
                if _above_dist < _below_dist * 0.7:
                    _nearest_liq_dir = 'LONG'
                elif _below_dist < _above_dist * 0.7:
                    _nearest_liq_dir = 'SHORT'

        direction, dir_size_mult, dir_details = resolve_direction(
            vol_regime, m9_score, m13_bias, m13_score, m13_details,
            m7_score=m7_score, m7_status=m7_status,
            swing_bias_1d=swing_bias, trend_dir=trend_dir, config=cfg,
            long_target_score=_long_tgt, short_target_score=_short_tgt,
            long_target_details=_long_det, short_target_details=_short_det,
            nearest_liq_direction=_nearest_liq_dir,
            m20_score=_m20_pre_score, m20_direction=_m20_pre_dir,
        )

        # Track M20 direction override
        if dir_details.get('m20_override'):
            stats['m20_override'] += 1

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
            _m13_status2, _m13_score2, m13_details = score_m13(
                df_1h, idx_1h, direction, df_15m, idx)
            if not _in_chop:
                m13_status = _m13_status2
                m13_score = _m13_score2
            # During chop: keep m13_score at 0.5 for ICS, but m13_details
            # is updated with direction-aware swing levels for M14

        # M1 + M2 still scored for ICS (not direction source)
        m1_dir, m1_score, _m1_details = score_m1(df_1h, idx_1h, cfg, df_15m=df_15m, idx_15m=idx)
        # Direction-aware scoring: flip M1 score when direction disagrees with trade
        if m1_dir == 'BEARISH' and direction == 'LONG':
            m1_score = 1.0 - m1_score
        elif m1_dir == 'BULLISH' and direction == 'SHORT':
            m1_score = 1.0 - m1_score
        m2_status, m2_score = score_m2(df_1h, df_2h, df_4h, df_1d, idx_1h, idx_2h, idx_4h, idx_1d)

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
            # Regime-aware threshold: relaxed in NEUTRAL/COMPRESSING
            # where weak trend is expected (Proposal 4)
            if vol_regime in ('NEUTRAL', 'COMPRESSING'):
                min_score = cfg.get('TREND_MIN_SCORE_NEUTRAL', cfg.get('TREND_MIN_SCORE', 0.05))
            else:
                min_score = cfg.get('TREND_MIN_SCORE', 0.05)
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

        # M4b: Intrabar CVD check (LucF-style) — aligned with scanner
        m4b_divergence = 'NONE'
        m4b_score = 0.5
        m4b_details = {}
        if cfg.get('M4B_INTRABAR_ENABLED', True):
            _m4b_found_bar = None
            for ci in range(max(0, idx - 24), idx + 1):
                d = df_15m['cvd_intrabar_div'].iloc[ci]
                if d != 'NONE':
                    m4b_divergence = d
                    _m4b_found_bar = ci
                    break
            if m4b_divergence != 'NONE':
                # Use score_intrabar_divergence for consistent scoring with scanner
                _intrabar_result = {
                    'divergence': m4b_divergence,
                    'bars_ago': idx - _m4b_found_bar if _m4b_found_bar is not None else 0,
                    'cvd_slope_12': 0.0,  # not available in backtest proxy
                }
                _m4b_status, m4b_score, m4b_details = score_intrabar_divergence(
                    _intrabar_result, direction)

        # Blend M4b into M4 if intrabar catches div that taker CVD missed
        # When M4 has no divergence but M4b does → use M4b's score (matches scanner behavior)
        m4_div_str = m4_div.get('layer_a_div', 'NONE') if isinstance(m4_div, dict) else 'NONE'
        if m4_div_str == 'NONE' and m4b_divergence != 'NONE':
            # M4 missed it, M4b caught it — apply M4b's signal
            m4_score = m4b_score
            if isinstance(m4_div, dict):
                m4_div['intrabar_div'] = m4b_divergence
                m4_div['intrabar_source'] = 'lucf_style'

        # M10 scored early — feeds into ICS pre-check (Proposal 1)
        m10_score = 0.5; m10_status = 'SKIP'; use_m10 = False
        if cfg.get('M10_ENABLED', False) and m10_data is not None:
            macro_row = m10_get_row(m10_data, ts)
            if macro_row:
                m10_status, m10_score, m10_details = score_m10_macro(macro_row, direction, trend_dir)
                use_m10 = True

        # ICS pre-check: M1-M4 + M10 (if available), before M5/M7/M8/M11-M14
        ics_pre, effective_floor = calc_ics(m1_score, m2_score, m3_score, m4_score, m4_status, 0.5,
                                            m10_score=m10_score, use_m10=use_m10, config=cfg)
        if m4_status == 'FAIL':
            stats['m4_false_anchored'] += 1
        # Use dedicated pre-check thresholds (lower than final ICS — fewer modules)
        precheck_floor = cfg.get('ICS_PRECHECK_FLOOR', cfg['ICS_FLOOR'])
        precheck_threshold = cfg.get('ICS_PRECHECK_THRESHOLD', cfg['ICS_THRESHOLD_NORMAL'])
        threshold = cfg['ICS_THRESHOLD_CAUTION'] if phase0_val >= 0.40 else precheck_threshold
        if is_summer:
            threshold += cfg.get('SUMMER_ICS_BOOST', 0)
        elif is_shoulder:
            threshold += cfg.get('SHOULDER_ICS_BOOST', 0)
        threshold += veto_soft_penalty
        if ics_pre < precheck_floor or ics_pre < threshold:
            stats['ics_blocked'] += 1
            continue

        # M5 (lazy, cached every 4 bars)
        m5_cache_key = idx // 4
        if not hasattr(run_backtest, '_m5_cache') or run_backtest._m5_cache_key != m5_cache_key:
            m5_status, m5_score, m5_details = score_m5(df_15m, idx, direction, cfg,
                n_bins=cfg['M5_VP_BINS'], lookback=cfg['M5_VP_LOOKBACK'],
                m13_details=m13_details)
            cascade = detect_cascade_setup(df_15m, idx)
            run_backtest._m5_cache = (m5_status, m5_score, m5_details, cascade)
            run_backtest._m5_cache_key = m5_cache_key
        else:
            m5_status, m5_score, m5_details, cascade = run_backtest._m5_cache

        # M5 Regime Gate — M5 is inverted in bull/ranging, predictive in bear/trending.
        # Cap M5 to neutral (0.5) when regime doesn't support it (forensic P1).
        if cfg.get('M5_REGIME_GATE_ENABLED', False):
            _m5_favorable_regimes = ('NEUTRAL', 'TRENDING', 'CHOP_MILD_BEAR')
            if vol_regime not in _m5_favorable_regimes:
                m5_score = 0.5  # neutralize — don't let inverted M5 affect ICS

        if m5_status == 'PASS':
            stats['m5_pass'] += 1
        else:
            stats['m5_fail'] += 1
        if cascade.get('cascade'):
            stats['cascade_detected'] += 1

        cascade_dir = m5_details.get('cascade_dir', 'NONE') if isinstance(m5_details, dict) else 'NONE'
        cascade_strength = m5_details.get('cascade_strength', 0.0) if isinstance(m5_details, dict) else 0.0

        # M7 already computed above (Phase 2)

        # M8
        if cfg.get('M8_ENABLED', False) and cached_funding_rate is not None:
            m8_status, m8_score, m8_details = score_m8_funding(cached_funding_rate, direction, cfg)
            use_m8 = True

        # M10 already scored above (before ICS pre-check)

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

        # M14: Sweep-Retest-Reclaim
        m14_score = 0.5; m14_status = 'SKIP'; m14_details = {}; use_m14 = False
        if cfg.get('M14_ENABLED', True):
            _swing_levels = m13_details.get('swing_lows', []) if direction == 'LONG' else m13_details.get('swing_highs', [])
            if _swing_levels:
                m14_status, m14_score, m14_details = score_m14(
                    df_15m, idx, direction, _swing_levels, config=cfg,
                    magnets=_magnets)
                if m14_status == 'PASS':
                    # Sweep detected + reclaimed → boost ICS
                    use_m14 = True
                elif m14_status == 'FAIL':
                    # Sweep detected but price sliced through → block entry
                    stats['m14_fail'] += 1
                    continue
                # SKIP (no sweep detected) → invisible, don't affect ICS

        # M17: Resistance Quality — validate nearest S/R level (aligned with scanner)
        m17_score = 0.5; m17_status = 'SKIP'; use_m17 = False
        if cfg.get('M17_ENABLED', True) and _sr:
            _current_price = float(row['Close'])
            _deriv_for_m17 = {}
            # Fetch derivatives summary for M17 defender analysis
            try:
                _deriv_for_m17 = get_derivatives_summary() if cfg.get('DERIV_ENABLED', False) else {}
                if 'error' in _deriv_for_m17:
                    _deriv_for_m17 = {}
            except Exception:
                _deriv_for_m17 = {}

            # Reuse volume profile from M5 cache (already computed)
            _m17_bc, _m17_vp = None, None
            if hasattr(run_backtest, '_vp_cache'):
                _m17_bc, _m17_vp = run_backtest._vp_cache

            if direction == 'LONG':
                _resistances = [sr for sr in _sr if sr[4] == 'RESISTANCE']
                if _resistances:
                    _nearest_res = min(_resistances, key=lambda x: abs(x[0] - _current_price))
                    _m17_result = score_resistance_quality(
                        _nearest_res[0], df_15m, idx, _m17_bc, _m17_vp,
                        _deriv_for_m17, 'LONG', config=cfg)
                    if _m17_result:
                        m17_score = _m17_result['composite']
                        m17_status = 'PASS'
                        use_m17 = True
                        stats['m17_pass'] += 1
                    else:
                        stats['m17_skip'] += 1
                else:
                    stats['m17_skip'] += 1
            elif direction == 'SHORT':
                _supports = [sr for sr in _sr if sr[4] == 'SUPPORT']
                if _supports:
                    _nearest_sup = min(_supports, key=lambda x: abs(x[0] - _current_price))
                    _m17_result = score_resistance_quality(
                        _nearest_sup[0], df_15m, idx, _m17_bc, _m17_vp,
                        _deriv_for_m17, 'SHORT', config=cfg)
                    if _m17_result:
                        m17_score = _m17_result['composite']
                        m17_status = 'PASS'
                        use_m17 = True
                        stats['m17_pass'] += 1
                    else:
                        stats['m17_skip'] += 1
                else:
                    stats['m17_skip'] += 1
            else:
                stats['m17_skip'] += 1
        else:
            stats['m17_skip'] += 1

        # Extract M4 divergence string early — needed by both veto and coherence
        m4_div_str_veto = m4_div.get('layer_a_div', 'NONE') if isinstance(m4_div, dict) else ('NONE' if m4_div is None else str(m4_div))

        # M2 Veto — aligned with scanner (after module scoring, before veto system)
        if cfg.get('M2_VETO_ENABLED', False):
            m2_veto_thresh = cfg.get('M2_VETO_THRESHOLD', 0.40)
            if direction == 'LONG' and m2_status == 'BEARISH' and m2_score < m2_veto_thresh:
                stats['m2_veto_block'] += 1
                continue
            if direction == 'SHORT' and m2_status == 'BULLISH' and m2_score < m2_veto_thresh:
                stats['m2_veto_block'] += 1
                continue

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
                # M4b is advisory only — not used for veto blocking
                m4_disagree = (direction == 'LONG' and m4_div_str_veto == 'BEARISH') or (direction == 'SHORT' and m4_div_str_veto == 'BULLISH')
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

        # M20: Failed Breakout Detector (scored for ICS)
        m20_score = 0.5; m20_status = 'SKIP'; m20_result = None; use_m20 = False
        if cfg.get('M20_ENABLED', True):
            try:
                _sr_for_m20 = _sr if _sr else None
                _mag_for_m20 = _magnets if _magnets else None
                m20_status, m20_score, m20_result = score_m20(
                    df_15m, idx, direction,
                    sr_levels=_sr_for_m20, magnets=_mag_for_m20,
                    config=cfg, atr_1h=_atr_1h)
                if m20_status == 'PASS':
                    stats['m20_pass'] += 1
                    use_m20 = True
                elif m20_status == 'FAIL':
                    stats['m20_fail'] += 1
                else:
                    stats['m20_skip'] += 1
            except Exception:
                stats['m20_skip'] += 1

        # ICS
        use_m13 = cfg.get('M13_ENABLED', False)
        ics, effective_floor = calc_ics(m1_score, m2_score, m3_score, m4_score, m4_status, m5_score,
                                        m7_score=m7_score, m8_score=m8_score, cross_asset_score=cross_asset_score,
                                        use_m7=use_m7, use_m8=use_m8, use_cross_asset=use_cross_asset,
                                        cascade_dir=cascade_dir, cascade_strength=cascade_strength,
                                        m9_score=m9_score, use_m9=use_m9, m10_score=m10_score, use_m10=use_m10,
                                        m11_score=m11_score, use_m11=use_m11, m12_score=m12_score, use_m12=use_m12,
                                        m13_score=m13_score, use_m13=use_m13, m14_score=m14_score, use_m14=use_m14,
                                        m17_score=m17_score, use_m17=use_m17,
                                        m20_score=m20_score, use_m20=use_m20,
                                        config=cfg)
        ics += gatekeeper.ics_boost

        # M5 sweet-spot boost — 0.3-0.5 bucket is the best performer
        # (Proposal 3: targeted boost, not blanket weight increase)
        m5_spot_low = cfg.get('M5_SWEET_SPOT_LOW', 0.30)
        m5_spot_high = cfg.get('M5_SWEET_SPOT_HIGH', 0.50)
        if m5_spot_low <= m5_score <= m5_spot_high:
            ics += cfg.get('M5_SWEET_SPOT_BOOST', 0.04)

        # ── Squeeze ICS boost (only when TRIGGERED + CONFIRMED, aligned with scanner) ──
        if squeeze_result['squeeze_status'] == 'TRIGGERED' and squeeze_confirmed and \
                squeeze_result.get('ics_boost', 0) > 0:
            ics += squeeze_result['ics_boost']

        # ── Squeeze regime override (aligned with scanner) ──
        # When squeeze is TRIGGERED + confirmed + overrides_regime, lift regime block
        # Note: In engine, M9 block happens early (line ~770). This override only helps
        # when regime is NOT in M9_BLOCK_REGIMES but is still unfavorable.
        _squeeze_overrode_regime = False
        if squeeze_result['squeeze_status'] == 'TRIGGERED' and squeeze_confirmed and \
                squeeze_result.get('overrides_regime', False):
            _squeeze_overrode_regime = True

        # ═══════════════════════════════════════════════════════════
        # COHERENCE CHECK — do module states tell a consistent story?
        # ═══════════════════════════════════════════════════════════
        coherence_penalty = 0.0
        if cfg.get('COHERENCE_CHECK_ENABLED', True):
            is_coherent, coherence_conflicts, coherence_penalty = check_coherence(
                direction, m4_div_str_veto, m5_details if isinstance(m5_details, dict) else {},
                m13_bias if not _in_chop else 'NEUTRAL', vol_regime,
                m7_score=m7_score, m2_status=m2_status, config=cfg,
            )
            if not is_coherent:
                stats['coherence_block'] += 1
                continue
            if coherence_penalty > 0:
                stats['coherence_penalty'] += 1

        # ═══════════════════════════════════════════════════════════
        # WICK RECLAIM — sweep-and-reclaim bonus/penalty
        # ═══════════════════════════════════════════════════════════
        wick_adj = 0.0
        if cfg.get('WICK_RECLAIM_ENABLED', True) and not _in_chop:
            _sr_levels = m13_details.get('swing_lows', []) if direction == 'LONG' else m13_details.get('swing_highs', [])
            if _sr_levels:
                wick_action, wick_adj, wick_details = detect_wick_reclaim(
                    df_15m, idx, direction, _sr_levels, config=cfg)
                if wick_action == 'RECLAIM':
                    stats['wick_reclaim_bonus'] += 1
                elif wick_action == 'SLICE_THROUGH':
                    if cfg.get('WICK_SLICE_BLOCK', False):
                        stats['wick_slice_block'] += 1
                        continue
                    stats['wick_reclaim_penalty'] += 1

        # Apply coherence + wick adjustments to ICS
        ics -= coherence_penalty
        ics += wick_adj

        # Track M14 stats
        if cfg.get('M14_ENABLED', True):
            if m14_status == 'PASS':
                stats['m14_pass'] += 1
            elif m14_status == 'FAIL':
                stats['m14_fail'] += 1
            else:
                stats['m14_skip'] += 1

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
            # ── M20 Direct Signal Path ──
            # When normal ICS fails but M20 detected a strong failed breakout
            # that overrode the direction, allow M20 to generate a signal directly.
            # Failed breakouts are high-conviction contrarian events that the normal
            # module pipeline doesn't score well (M3/M4/M13 work against the flip).
            m20_direct_threshold = cfg.get('M20_DIRECT_SIGNAL_THRESHOLD', 0.85)
            m20_override_active = dir_details.get('m20_override') is not None
            _m20_direct = (m20_status == 'PASS' and m20_score >= m20_direct_threshold and
                           m20_result and m20_result.get('status') == 'FAILED' and
                           m20_override_active)
            if _m20_direct:
                # M20 strong failed breakout — bypass ICS gate
                stats['m20_direct_signal'] += 1
                stats['m20_ics_bypass'] += 1
                # Continue to trade level computation below (don't skip)
            else:
                stats['ics_blocked'] += 1
                continue
        else:
            _m20_direct = False

        if ics > cfg.get('ICS_CEILING', 1.0):
            stats['ics_ceiling_skip'] += 1
            continue

        if m4_status == 'FAIL' and not _m20_direct:
            stats['m4_required_skip'] += 1
            continue

        # Bias gate
        bad_gate_months = [3, 7, 9]
        if cfg.get('BIAS_GATE_ENABLED', False) and direction == 'LONG' and swing_bias == 'BEARISH' and ts.month in bad_gate_months:
            if ics < cfg.get('BIAS_GATE_LONG_ICS', 0.65):
                stats['bias_gate_skip'] += 1
                continue

        passed, reason = check_entry_filters(df_15m, idx, direction, swing_bias, phase0_val, atr_1h, config=cfg)
        if not passed and not _m20_direct:
            stats['filter_blocked'] += 1
            continue

        # ═══════════════════════════════════════════════════════════
        # SQUEEZE DETECTION (M18) — Run per bar, track signals
        # ═══════════════════════════════════════════════════════════
        squeeze_result = {'squeeze_type': 'NONE', 'squeeze_status': 'NONE'}
        squeeze_confirmed = False
        squeeze_filters = {}
        breakout_result = None
        if squeeze_enabled and idx >= 47:
            _close_val_sq = float(row['Close'])
            _vol_ma20_sq = float(row['vol_ma20']) if row['vol_ma20'] > 0 else 1.0
            _vol_ratio_sq = float(row['Volume'] / _vol_ma20_sq)
            _taker_sq = float(row['taker_ratio'])
            _bar_range_sq = (float(row['High']) - float(row['Low'])) / _close_val_sq * 100 if _close_val_sq > 0 else 0.5

            # Bar-level ignition (matches scanner computation)
            if idx >= 19:
                _bar_range_ma_sq = float(df_15m['High'].iloc[idx-19:idx+1].sub(
                    df_15m['Low'].iloc[idx-19:idx+1]).div(
                    df_15m['Close'].iloc[idx-19:idx+1]).mean() * 100)
            else:
                _bar_range_ma_sq = _bar_range_sq
            _bar_range_expansion = _bar_range_sq / _bar_range_ma_sq if _bar_range_ma_sq > 0 else 1.0

            # OI proxy: volume accumulation relative to average
            if idx > 67:
                _vol_cumsum_48 = float(df_15m['Volume'].iloc[idx-47:idx+1].sum())
                _vol_cumsum_ma = float(df_15m['Volume'].iloc[idx-67:idx+1].rolling(20).mean().iloc[-1])
                _oi_proxy = _vol_cumsum_48 / _vol_cumsum_ma if _vol_cumsum_ma > 0 else 1.0
            else:
                _oi_proxy = 1.0

            # VWAP distance
            _vwap_sq = float(df_15m['vwap'].iloc[idx]) if 'vwap' in df_15m.columns else _close_val_sq
            _vwap_dist_sq = (_close_val_sq - _vwap_sq) / _vwap_sq * 100 if _vwap_sq > 0 else 0

            # Squeeze quality (matches scanner formula)
            _rw_score = max(0, min(1, 1 - (squeeze_compression_history[-1][0] - 1.5) / 4.0)) if squeeze_compression_history else 0.5
            _vr_score = max(0, min(1, 1 - (_vol_ratio_sq - 0.05) / 0.20))
            _oip_score = max(0, min(1, (_oi_proxy - 0.7) / 0.5))
            _vd_score = max(0, min(1, 1 - abs(_vwap_dist_sq) / 1.0))
            _squeeze_quality = _rw_score * 0.30 + _vr_score * 0.25 + _oip_score * 0.25 + _vd_score * 0.20

            sq_result = {
                'price': _close_val_sq,
                'atr': float(row['atr']),
                'range_width': squeeze_compression_history[-1][0] if squeeze_compression_history else 5.0,
                'raw_taker_ratio': _taker_sq,
                'raw_bar_range_pct': _bar_range_sq,
                'vol_trend': _vol_ratio_sq,
                'vol_ratio': float(row.get('vol_ratio', 1.0)),
                'vol_ma20': _vol_ma20_sq,
                'bar_vol_spike': _vol_ratio_sq,
                'bar_range_expansion': _bar_range_expansion,
                'bar_taker_extreme': _taker_sq > 0.65 or _taker_sq < 0.35,
                'oi_proxy': _oi_proxy,
                'vwap_dist': _vwap_dist_sq,
                'squeeze_quality': _squeeze_quality,
                'rsi': float(df_15m['rsi'].iloc[idx]) if 'rsi' in df_15m.columns and not pd.isna(df_15m['rsi'].iloc[idx]) else 50.0,
            }
            # Pass magnets + sr_levels for consistent TP/SL with scanner
            squeeze_result = detect_squeeze(
                sq_result, config=cfg,
                last_signal_bar=squeeze_last_squeeze_bar,
                current_bar=idx,
                compression_history=squeeze_compression_history,
                df_15m=df_15m.iloc[:idx+1],
                magnets=_magnets if _magnets else None,
                sr_levels=_sr if _sr else None,
                liq_levels=None,  # live order book not available in backtest
            )
            if squeeze_result['squeeze_type'] != 'NONE':
                stats['squeeze_detected'] += 1
                if squeeze_result.get('failed_breakout'):
                    stats['squeeze_failed_breakout'] += 1
                if squeeze_result['squeeze_status'] == 'TRIGGERED':
                    stats['squeeze_triggered'] += 1
                    squeeze_last_squeeze_bar = idx
                elif squeeze_result['squeeze_status'] == 'PENDING':
                    stats['squeeze_pending'] += 1

            # ── Squeeze 5-filter confirmation gate (aligned with scanner) ──
            sq_type = squeeze_result.get('squeeze_type', 'NONE')
            sq_dir = squeeze_result.get('direction', 'NEUTRAL')
            if sq_type != 'NONE' and sq_dir != 'NEUTRAL':
                # Filter 1: EMA trend — regime-adaptive
                if cfg.get('SQUEEZE_CONFIRM_EMA', True) and len(df_15m) >= 55:
                    _close = df_15m['Close'].iloc[:idx+1]
                    _ema21 = float(_close.ewm(span=21, adjust=False).mean().iloc[-1])
                    _ema55 = float(_close.ewm(span=55, adjust=False).mean().iloc[-1])
                    _ema_spread = (_ema21 - _ema55) / _ema55 * 100 if _ema55 > 0 else 0
                    _ema_trend = 'BULL' if _ema21 > _ema55 else 'BEAR'
                    _contrarian = (sq_dir == 'LONG' and _ema_trend == 'BEAR') or \
                                  (sq_dir == 'SHORT' and _ema_trend == 'BULL')
                    _aligned = not _contrarian
                    if _ema_spread < 0:
                        squeeze_filters['ema_regime'] = _contrarian
                    else:
                        squeeze_filters['ema_regime'] = _aligned
                else:
                    squeeze_filters['ema_regime'] = True

                # Filter 2: CVD divergence agrees
                if cfg.get('SQUEEZE_CONFIRM_CVD', True):
                    _m4b_div = m4b_divergence
                    squeeze_filters['cvd_agrees'] = not ((sq_dir == 'LONG' and _m4b_div == 'BEARISH') or
                                                         (sq_dir == 'SHORT' and _m4b_div == 'BULLISH'))
                else:
                    squeeze_filters['cvd_agrees'] = True

                # Filter 3: RSI not extreme against direction
                if cfg.get('SQUEEZE_CONFIRM_RSI', True) and 'rsi' in df_15m.columns:
                    _rsi = float(df_15m['rsi'].iloc[idx]) if not pd.isna(df_15m['rsi'].iloc[idx]) else 50
                    squeeze_filters['rsi_ok'] = (sq_dir == 'LONG' and _rsi < 75) or \
                                                 (sq_dir == 'SHORT' and _rsi > 25)
                else:
                    squeeze_filters['rsi_ok'] = True

                # Filter 4: Quality score >= 0.5
                if cfg.get('SQUEEZE_CONFIRM_QUALITY', True):
                    squeeze_filters['quality_high'] = squeeze_result.get('squeeze_score', 0) >= 0.5
                else:
                    squeeze_filters['quality_high'] = True

                # Filter 5: ATR floor
                if cfg.get('SQUEEZE_CONFIRM_ATR_FLOOR', True):
                    _atr_now = float(df_15m['atr'].iloc[idx]) if not pd.isna(df_15m['atr'].iloc[idx]) else 0
                    _atr_hard_floor = cfg.get('SQUEEZE_MIN_ATR', 5.0)
                    _atr_lookback = cfg.get('SQUEEZE_ATR_LOOKBACK', 8640)
                    _atr_pctile = cfg.get('SQUEEZE_ATR_FLOOR_PCTILE', 15)
                    _atr_series = df_15m['atr'].iloc[:idx+1].dropna()
                    if len(_atr_series) > 100:
                        _atr_window = _atr_series.iloc[-min(len(_atr_series), _atr_lookback):]
                        _atr_threshold = float(np.percentile(_atr_window, _atr_pctile))
                    else:
                        _atr_threshold = _atr_hard_floor
                    _atr_effective = max(_atr_hard_floor, _atr_threshold)
                    squeeze_filters['atr_floor'] = _atr_now >= _atr_effective
                else:
                    squeeze_filters['atr_floor'] = True

                squeeze_confirmed = all(squeeze_filters.values())
                if squeeze_confirmed:
                    stats['squeeze_confirmed'] += 1
                else:
                    stats['squeeze_not_confirmed'] += 1

            # ── M19: Breakout Confirmation (aligned with scanner) ──
            if sq_type != 'NONE' and sq_dir != 'NEUTRAL':
                # Build a minimal result dict for check_breakout_filters
                # (same structure scanner passes)
                _bc_result = {
                    'squeeze': squeeze_result,
                    'm4': {'div': m4_div},
                    'm4b': {'divergence': m4b_divergence},
                    'm10': {'details': m10_details} if m10_details else {},
                    'vol_trend': float(row['Volume'] / row['vol_ma20']) if row['vol_ma20'] > 0 else 1.0,
                    'bar_vol_spike': float(row['Volume'] / row['vol_ma20']) if row['vol_ma20'] > 0 else 1.0,
                    'derivatives': _deriv_for_m17 if '_deriv_for_m17' in locals() else {},
                    'magnets': [(p, s, False, None) for p, s, *_ in _magnets] if _magnets else [],
                    'price': float(row['Close']),
                    'raw_taker_ratio': float(row['taker_ratio']),
                    'exchange_activity': {},  # not available in backtest
                }
                # Add OI roc from derivatives if available
                if _bc_result['derivatives']:
                    _bc_result['derivatives']['oi_roc_1h'] = _bc_result['derivatives'].get('oi_roc_1h', 0)
                    _bc_result['derivatives']['funding_rate'] = _bc_result['derivatives'].get('funding_rate', 0)

                breakout_result = check_breakout_filters(
                    _bc_result, df_15m=df_15m.iloc[:idx+1], config=cfg)

                if breakout_result['status'] == 'CONFIRMED':
                    stats['breakout_confirmed'] += 1
                elif breakout_result['status'] == 'WEAK':
                    stats['breakout_weak'] += 1
                elif breakout_result['status'] == 'REJECTED':
                    stats['breakout_rejected'] += 1
                    # If squeeze was TRIGGERED but breakout REJECTED, suppress
                    if squeeze_result['squeeze_status'] == 'TRIGGERED':
                        squeeze_confirmed = False

        # ── Squeeze Entry Gate ──
        # When a squeeze is detected, only allow entry if:
        #   - squeeze is TRIGGERED (entry condition already met), OR
        #   - squeeze entry_triggered is True (entry condition met this bar)
        # This aligns the engine with the scanner's squeeze entry logic,
        # preventing entries at the current price when the squeeze is still
        # PENDING (waiting for coil breakdown).
        _sq_type = squeeze_result.get('squeeze_type', 'NONE')
        _sq_status = squeeze_result.get('squeeze_status', 'NONE')
        _sq_entry_triggered = squeeze_result.get('entry_triggered', True)
        if _sq_type != 'NONE' and _sq_status == 'PENDING' and not _sq_entry_triggered:
            stats['squeeze_direction_block'] += 1
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

        # Position Sizing — base size * direction resolver multiplier
        size = cfg['SIZE_STD'] * dir_size_mult
        transition_range = cfg.get('TRANSITION_SCORE_RANGE', 0.20)
        is_transition = abs(trend_val) < transition_range

        # Track M14 stats
        if cfg.get('M14_ENABLED', True):
            if m14_status == 'PASS':
                stats['m14_pass'] += 1
            elif m14_status == 'FAIL':
                stats['m14_fail'] += 1
            else:
                stats['m14_skip'] += 1

        # M14 Sweep Gate (bypass for M20 direct signals)
        sweep_passed, sweep_reason = check_sweep_gate(m14_status, m14_score, cfg)
        if not sweep_passed and not _m20_direct:
            stats['filter_blocked'] += 1
            continue

        # Entry
        entry_price = row['Close']
        atr_for_sl = atr_1h if not pd.isna(atr_1h) else row['atr']

        # ── Liquidity-Aware SL/TP ──
        # Build seasonal config overrides for ATR fallback
        _seasonal_cfg = dict(cfg)
        if _in_chop:
            _seasonal_cfg['SL_ATR_STD'] = cfg.get('CHOP_SL_ATR', 1.0)
            _seasonal_cfg['SL_HARD_MAX_PCT'] = cfg.get('CHOP_SL_HARD_MAX', 0.012)
            _seasonal_cfg['TP1_ATR'] = cfg.get('CHOP_TP1_ATR', 0.6)
        elif is_transition:
            _seasonal_cfg['SL_ATR_STD'] = cfg.get('SL_ATR_TRANSITION', 1.0)
            _seasonal_cfg['TP1_ATR'] = cfg.get('TP1_ATR_TRANSITION', 1.2)
        elif is_summer:
            _seasonal_cfg['SL_ATR_STD'] = cfg.get('SL_ATR_STD_SUMMER', cfg['SL_ATR_STD'])
            _seasonal_cfg['SL_HARD_MAX_PCT'] = cfg.get('SL_HARD_MAX_SUMMER', cfg['SL_HARD_MAX_PCT'])
            _seasonal_cfg['TP1_ATR'] = cfg.get('TP1_ATR_SUMMER', cfg['TP1_ATR'])
        elif is_shoulder:
            _seasonal_cfg['SL_ATR_STD'] = cfg.get('SHOULDER_SL_ATR', cfg['SL_ATR_STD'])
            _seasonal_cfg['SL_HARD_MAX_PCT'] = cfg.get('SHOULDER_SL_HARD_MAX', cfg['SL_HARD_MAX_PCT'])
            _seasonal_cfg['TP1_ATR'] = cfg.get('SHOULDER_TP1_ATR', cfg['TP1_ATR'])

        # Gather liquidity data for level placement
        _magnets_for_levels = []
        _sr_for_levels = []
        if cfg.get('LIQUIDITY_LEVELS_ENABLED', True):
            from src.modules.m5_liquidation import build_volume_profile, find_magnets, find_support_resistance
            _highs = df_15m['High'].values[:idx+1].astype(float)
            _lows = df_15m['Low'].values[:idx+1].astype(float)
            _closes = df_15m['Close'].values[:idx+1].astype(float)
            _volumes = df_15m['Volume'].values[:idx+1].astype(float)
            _bc, _vp, _ = build_volume_profile(
                _highs, _lows, _closes, _volumes,
                n_bins=cfg.get('M5_VP_BINS', 50), lookback=cfg.get('M5_VP_LOOKBACK', 672))
            if _bc is not None:
                _magnets_for_levels = find_magnets(_bc, _vp)
            _sr_for_levels = find_support_resistance(df_15m, idx)

        trade_levels = calc_trade_levels(
            entry_price, direction, atr_for_sl,
            row.get('vol_ratio', np.nan),
            magnets=_magnets_for_levels,
            sr_levels=_sr_for_levels,
            liq_levels=None,  # M15 liq levels not cached in engine yet
            cfg=_seasonal_cfg,
        )

        sl = trade_levels['sl']
        tp1 = trade_levels['tp1']
        tp2 = trade_levels['tp2']
        tp3 = trade_levels['tp3']
        sl_dist = abs(entry_price - sl)

        # ═══════════════════════════════════════════════════════════
        # LIQUIDITY-AWARE TP — adjust TP fractions based on volume terrain
        # ═══════════════════════════════════════════════════════════
        if cfg.get('LIQUIDITY_TP_ENABLED', True):
            # Cache volume profile alongside M5 (same cadence)
            vp_cache_key = idx // 4
            if not hasattr(run_backtest, '_vp_cache') or run_backtest._vp_cache_key != vp_cache_key:
                _highs = df_15m['High'].values.astype(float)
                _lows = df_15m['Low'].values.astype(float)
                _closes = df_15m['Close'].values.astype(float)
                _volumes = df_15m['Volume'].values.astype(float)
                from src.modules.m5_liquidation import build_volume_profile
                _vp_centers, _vp_profile, _vp_edges = build_volume_profile(
                    _highs[:idx+1], _lows[:idx+1], _closes[:idx+1], _volumes[:idx+1],
                    n_bins=cfg.get('M5_VP_BINS', 50), lookback=cfg.get('M5_VP_LOOKBACK', 672))
                run_backtest._vp_cache = (_vp_centers, _vp_profile)
                run_backtest._vp_cache_key = vp_cache_key
            else:
                _vp_centers, _vp_profile = run_backtest._vp_cache

            liq_tp = compute_liquidity_aware_tp(
                entry_price, tp1, tp2, tp3, direction,
                _vp_centers, _vp_profile, config=cfg)

            stop_risk = compute_stop_risk(
                entry_price, sl, direction,
                _vp_centers, _vp_profile, config=cfg)

            # Apply adjusted TP fractions
            tp1_close_frac = liq_tp['tp1_close_frac']
            tp2_close_frac = liq_tp['tp2_close_frac']

            # Tighten stop if liquidity grab risk detected
            if stop_risk['has_stop_risk'] and cfg.get('STOP_RISK_TIGHTEN', True):
                tighten_factor = cfg.get('STOP_RISK_TIGHTEN_FACTOR', 0.85)
                sl_dist_tightened = sl_dist * tighten_factor
                if direction == 'LONG':
                    sl = entry_price - sl_dist_tightened
                else:
                    sl = entry_price + sl_dist_tightened
        else:
            tp1_close_frac = cfg['TP1_CLOSE']
            tp2_close_frac = cfg['TP2_CLOSE']

        # Chop regime override: exit fully at TP1, no TP2/TP3 continuation
        if _in_chop:
            tp1_close_frac = cfg.get('CHOP_TP1_CLOSE', 0.90)
            tp2_close_frac = 0.0  # no remaining for TP2/TP3

        trade = Trade(ts, direction, entry_price, sl, tp1, tp2, tp3, size,
                      m1_dir, m2_status, m3_score, m4_status, m5_status, m5_score,
                      ics, phase0_val,
                      f"M9={vol_regime} M13={m13_bias} M1={m1_dir} M2={m2_status} M3={m3_status} M4={m4_status} M4b={m4b_divergence} M5={m5_status} ICS={ics:.3f}",
                      m1_score=m1_score, m2_score=m2_score, m4_score=m4_score, m7_score=m7_score,
                      m8_score=m8_score, m8_status=m8_status, m9_score=m9_score, m9_status=m9_status,
                      m10_score=m10_score, m10_status=m10_status, m11_score=m11_score, m11_status=m11_status,
                      m12_score=m12_score, m12_status=m12_status,
                      m13_score=m13_score, m13_status=m13_status,
                      m14_score=m14_score, m14_status=m14_status,
                      vol_regime=vol_regime, trend_dir=trend_dir, trend_val=trend_val,
                      cross_asset_score=cross_asset_score, session_name=session_name,
                      veto_soft_penalty=veto_soft_penalty, gatekeeper_passed=gatekeeper.passed,
                      m7_details=m7_details,
                      tp1_close_frac=tp1_close_frac, tp2_close_frac=tp2_close_frac,
                      squeeze_type=squeeze_result.get('squeeze_type', 'NONE'),
                      squeeze_score=squeeze_result.get('squeeze_score', 0.0),
                      squeeze_strong=squeeze_result.get('squeeze_strong', False),
                      squeeze_trigger_type=squeeze_result.get('trigger_type', 'NONE'),
                      squeeze_failed_breakout=squeeze_result.get('failed_breakout', False),
                      squeeze_box_type=squeeze_result.get('box_type', 'UNKNOWN'),
                      squeeze_lifecycle=squeeze_result.get('lifecycle_stage', 'NONE'))
        trade.dir_size_mult = dir_size_mult
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
