"""
JIMI Framework — Configuration Loader
Loads settings from YAML, falls back to built-in defaults.
Usage:
    from src.config import CONFIG
    # or
    from src.config import load_config
    cfg = load_config("config/v615.yaml")
"""

import os
import yaml

_DEFAULTS = {
    # Gate thresholds
    "ICS_THRESHOLD_NORMAL": 0.50,
    "ICS_THRESHOLD_CAUTION": 0.52,
    "ICS_FLOOR": 0.50,
    "ICS_FLOOR_M4_FALSE": 0.50,
    "ICS_CEILING": 0.70,
    # Module weights
    "M1_WEIGHT": 0.08,
    "M2_WEIGHT": 0.00,
    "M3_WEIGHT": 0.22,
    "M4_WEIGHT": 0.38,
    "M5_WEIGHT": 0.25,
    "M6_WEIGHT": 0.10,
    "M7_WEIGHT": 0.00,
    "M8_WEIGHT": 0.10,
    "M9_WEIGHT": 0.00,
    "M10_WEIGHT": 0.10,
    "M11_WEIGHT": 0.12,
    "M12_WEIGHT": 0.05,
    # v6.12
    "BIAS_GATE_ENABLED": True,
    "BIAS_GATE_LONG_ICS": 0.65,
    "TREND_FILTER_ENABLED": True,
    "TREND_BLOCK_COUNTER_TREND": False,
    "TREND_STRONG_ONLY": False,
    "TREND_MIN_SCORE": 0.15,
    "MONTHLY_DD_CIRCUIT": 0.05,
    # v6.13 Seasonal
    "SUMMER_MONTHS": [6, 7, 8, 9],
    "SUMMER_SIZE_MULT": 0.60,
    "SUMMER_ICS_BOOST": 0.03,
    "TP1_CLOSE_BASE": 0.30,
    "TP1_CLOSE_SUMMER": 0.45,
    "TP1_ATR_SUMMER": 0.7,
    "MAX_CONSEC_LOSS_SUMMER": 2,
    "CONSEC_LOSS_PAUSE_SUMMER": 12,
    "PHASE0_SUMMER_BLOCK": 0.60,
    "SHOULDER_MONTHS": [3, 10],
    "SHOULDER_SIZE_MULT": 0.50,
    "SHOULDER_SL_ATR": 1.4,
    "SHOULDER_SL_HARD_MAX": 0.016,
    "SHOULDER_TP1_ATR": 0.8,
    "SHOULDER_TP1_CLOSE": 0.45,
    "SHOULDER_ICS_BOOST": 0.02,
    "SHOULDER_MAX_TRADES_DAY": 3,
    # Module gates
    "M2_VETO_ENABLED": True,
    "M2_VETO_THRESHOLD": 0.40,
    "M5_FAIL_ICS_BOOST": 0.06,
    "M5_FAIL_HARD_THRESHOLD": 0.25,
    "M7_HARD_GATE": False,
    "M7_GATE_THRESHOLD": 0.30,
    "M7_GATE_STRONG_THRESHOLD": 0.60,
    "M7_GATE_STRONG_BOOST": 0.04,
    "SESSION_ASIAN_BLOCK": False,
    "POST_CRASH_COOLDOWN": True,
    "POST_CRASH_THRESHOLD": 0.10,
    "POST_CRASH_BARS": 48,
    "CASCADE_MULTIPLIER": 1.12,
    "CASCADE_PENALTY": 0.85,
    "DIR_VETO_ENABLED": True,
    # Indicators
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "EMA_FAST": 21,
    "EMA_SLOW": 55,
    "RSI_PERIOD": 14,
    "ATR_PERIOD": 14,
    # M3 VWAP
    "VWAP_LOOKBACK": 96,
    "VWAP_ZONE_PCT": 0.012,
    "VOL_THRESHOLD": 0.25,
    "TAKER_LONG": 0.52,
    "TAKER_SHORT": 0.48,
    "TAKER_FILLNA": 0.50,
    "SIGNAL_EXPIRY": 3,
    # M4 CVD
    "CVD_LOOKBACK": 36,
    "CVD_DIVERGENCE_WINDOW": 12,
    "M4_ZL_LOOKBACK": 18,
    "M4_ZL_MOMENTUM_BARS": 8,
    "M4_DIV_WEIGHT": 0.40,
    "M4_ZL_WEIGHT": 0.60,
    # M5 Liquidation
    "M5_VP_LOOKBACK": 672,
    "M5_VP_BINS": 50,
    "M5_MIN_SCORE": 0.25,
    # Stop loss
    "SL_ATR_STD": 1.3,
    "SL_ATR_STD_SUMMER": 1.6,
    "SL_ATR_TRANSITION": 1.0,
    "TP1_ATR_TRANSITION": 1.2,
    "TRANSITION_SIZE_MULT": 0.50,
    "TRANSITION_SCORE_RANGE": 0.20,
    "SL_HARD_MAX_PCT": 0.018,
    "SL_HARD_MAX_SUMMER": 0.018,
    "SL_BREAKEVEN_AFTER_TP1": True,
    "EARLY_EXIT_BARS": 16,
    "EARLY_EXIT_MIN_LOSS": 0.003,
    "EARLY_EXIT_BARS_SUMMER": 12,
    "EARLY_EXIT_MIN_LOSS_SUMMER": 0.002,
    # Take profit
    "TP1_ATR": 1.5,
    "TP2_ATR": 2.0,
    "TP3_ATR": 3.5,
    "TP1_CLOSE": 0.30,
    "TP2_CLOSE": 0.30,
    # Position sizing
    "SIZE_STD": 5.0,
    "SIZE_LONG": 5.0,
    "SIZE_M2_NEUTRAL": 2.5,
    "SIZE_CAUTION": 3.5,
    # Entry filters
    "BAR_MOVE_ATR": 0.5,
    "ATR_FILTER_MAX": 0.035,
    "MIN_ENTRY_DIST_PCT": 0.002,
    # Risk management
    "MAX_TRADES_DAY": 5,
    "MAX_TRADES_DAY_SUMMER": 3,
    "MAX_DAILY_LOSS": 0.05,
    "MAX_DAILY_LOSS_SUMMER": 0.03,
    "COOLDOWN_BARS": 2,
    "COOLDOWN_BARS_SUMMER": 4,
    "SHOULDER_COOLDOWN_BARS": 3,
    "ROLLING_WR_WINDOW": 20,
    "ROLLING_WR_MIN": 0.15,
    "MAX_CONSEC_LOSS": 3,
    "CONSEC_LOSS_PAUSE_BARS": 8,
    "LONG_MIN_ICS": 0.55,
    # M6 Derivatives
    "M3_WEIGHT_DERIV": 0.25,
    "M4_WEIGHT_DERIV": 0.20,
    "DERIV_ENABLED": True,
    # M7
    "M7_ENABLED": True,
    "M7_SIZE_REDUCTION": 0.70,
    "M7_SIZE_MILD": 0.85,
    # M8
    "M8_ENABLED": True,
    "M8_HIGH_FUNDING": 0.05,
    "M8_LOW_FUNDING": -0.05,
    "M8_FLIP_BONUS": 0.15,
    # M9
    "M9_ENABLED": True,
    "M9_BLOCK_REGIMES": ["CRISIS"],
    "M9_SIZE_CHOP": 0.50,
    "M9_SIZE_COMPRESSING": 0.80,
    # M10
    "M10_ENABLED": True,
    "M10_FETCH_ON_BACKTEST": True,
    # M11
    "M11_ENABLED": True,
    "M11_RSI_PERIOD_15M": 14,
    "M11_RSI_PERIOD_1H": 14,
    "M11_REQUIRE_AGREEMENT": True,
    # M12
    "M12_ENABLED": True,
    "M12_LIVE_ONLY": True,
    # M13 Structure (HTF swing direction)
    "M13_ENABLED": True,
    "M13_WEIGHT": 0.10,
    "M13_DEFER_IN_CHOP": True,
    # Coherence check
    "COHERENCE_CHECK_ENABLED": True,
    "COHERENCE_MAX_CONFLICTS": 3,
    "COHERENCE_M4_PENALTY": 0.04,
    "COHERENCE_M5_PENALTY": 0.03,
    "COHERENCE_M13_PENALTY": 0.03,
    "COHERENCE_M7_PENALTY": 0.02,
    "COHERENCE_MAX_PENALTY": 0.10,
    # Liquidity-aware TP
    "LIQUIDITY_TP_ENABLED": True,
    "LIQUIDITY_FRICTION_THRESHOLD": 0.60,
    "LIQUIDITY_VOID_THRESHOLD": 0.20,
    "LIQUIDITY_TP1_ADJUST_UP": 0.08,
    "LIQUIDITY_TP1_ADJUST_DOWN": 0.05,
    "STOP_RISK_THRESHOLD": 0.55,
    "STOP_RISK_TIGHTEN": True,
    "STOP_RISK_TIGHTEN_FACTOR": 0.85,
    # Adaptive direction
    "ADAPTIVE_DIR_ENABLED": True,
    "ADAPTIVE_DIR_MIN_BIAS": 0.10,
    "ADAPTIVE_DIR_BLOCK_THRESHOLD": 0.60,
    # Data freshness
    "DATA_FRESHNESS_ENABLED": True,
    "DATA_FRESHNESS_MAX_AGE_MIN": 20,
    "DATA_FRESHNESS_CHECK_INTERVAL": 5,
    # Veto system
    "VETO_ENABLED": True,
    "VETO_CRISIS_HARD": True,
    "VETO_CHOP_HARD": False,
    "VETO_STALE_DATA_HARD": True,
    "VETO_MONTHLY_DD_HARD": True,
    "VETO_DIR_CONFLICT_HARD": True,
    # Adaptive weights
    "ADAPTIVE_WEIGHTS_ENABLED": True,
    "ADAPTIVE_DECAY": 0.97,
    "ADAPTIVE_MIN_MULT": 0.8,
    "ADAPTIVE_MAX_MULT": 1.2,
    "ADAPTIVE_WARMUP_TRADES": 15,
    # Session awareness
    "SESSION_AWARENESS_ENABLED": True,
    "SESSION_ASIAN_MULT": 0.85,
    "SESSION_EU_MULT": 1.0,
    "SESSION_US_MULT": 1.05,
    "SESSION_LATE_US_MULT": 0.90,
    "SESSION_US_OPEN_BOOST": 1.10,
    # Multi-TF confirmation
    "MTF_CONFIRM_ENABLED": True,
    "MTF_1H_CANDLE_CHECK": True,
    "MTF_4H_EMA_CHECK": False,
    # M1 enhancement
    "M1_RSI_ENABLED": True,
    "M1_MOMENTUM_ENABLED": True,
    "M1_RSI_PERIOD": 14,
    "M1_RSI_OVERBOUGHT": 70,
    "M1_RSI_OVERSOLD": 30,
    "M1_MOMENTUM_LOOKBACK": 6,
    # Cross-asset
    "CROSS_ASSET_ENABLED": True,
    "CROSS_ASSET_BTC_WEIGHT": 0.08,
    "CROSS_ASSET_LOOKBACK": 48,
    # Data
    "PAIR": "ETHUSDT",
    "WARMUP_BARS_1H": 168,
}


def load_config(path=None):
    """Load config from YAML file, merged with defaults."""
    cfg = dict(_DEFAULTS)
    if path and os.path.exists(path):
        with open(path) as f:
            user = yaml.safe_load(f) or {}
            cfg.update(user)
    return cfg


# Singleton — importable as `from src.config import CONFIG`
CONFIG = load_config(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.yaml")
)
