"""
M55: US Treasury Auction Session Bias (Regime-Conditional)

On US 10Y/30Y Treasury auction result days (~2x/week, 17:00 UTC = 01:00 MYT),
applies a session-conditional directional bias based on:
  - Wyckoff phase (M21)
  - Volatility regime (M9)
  - Demand signal: STRONG_DEMAND / GOOD_DEMAND / NORMAL / WEAK_DEMAND
  - Auction type: 10Y / 30Y

Backtested on 200 Treasury auctions (2018-2026) against ETH/USDT 15m data.

Key findings (24h return):
  24h aggregate: +0.063% avg, 47.5% win, n=200 — NOISE (p=0.87)
  Treasury auctions are NOISE at the 24h level — only regime combos have edge.

  WEAK_DEMAND:               -3.479% avg, 42.9% win, n=14 → SHORT
  MARKUP × CRISIS × GOOD:    +2.241% avg, 82% win, n=11  → LONG
  MARKUP × NEUTRAL × NORMAL: +1.905% avg, 67% win, n=9   → LONG
  MARKUP × CRISIS × STRONG:  +5.817% avg, 100% win, n=3  → LONG (small n)

  Transmission chain: NO persistence across any session transition.

Thesis (user #39):
  Weak demand → tail widens → 10Y yield spike → risk-off → ETH sell-off.
  Yield spike → DXY strength → challenging for Asia open.
  Weekly series: trend in debt demand monitored over consecutive sessions.

Usage:
    from src.modules.m55_treasury_auction import score_m55_treasury, format_m55
"""

from datetime import datetime, timedelta
import json
import os

# Auction data: most recent ~6 months hardcoded; rest fetched live
# Format: {date: {'type': '10Y'|'30Y', 'bid_to_cover': float, 'tail_bps': float}}

RECENT_AUCTIONS = {
    '2025-10-08': {'type': '10Y', 'bid_to_cover': 2.40, 'tail_bps': 0.9},
    '2025-10-09': {'type': '30Y', 'bid_to_cover': 2.20, 'tail_bps': 1.3},
    '2025-11-05': {'type': '10Y', 'bid_to_cover': 2.48, 'tail_bps': 0.4},
    '2025-11-06': {'type': '30Y', 'bid_to_cover': 2.30, 'tail_bps': 0.5},
    '2025-12-10': {'type': '10Y', 'bid_to_cover': 2.53, 'tail_bps': 0.2},
    '2025-12-11': {'type': '30Y', 'bid_to_cover': 2.35, 'tail_bps': 0.3},
    '2026-01-14': {'type': '10Y', 'bid_to_cover': 2.49, 'tail_bps': 0.4},
    '2026-01-15': {'type': '30Y', 'bid_to_cover': 2.31, 'tail_bps': 0.6},
    '2026-02-11': {'type': '10Y', 'bid_to_cover': 2.44, 'tail_bps': 0.7},
    '2026-02-12': {'type': '30Y', 'bid_to_cover': 2.25, 'tail_bps': 0.9},
    '2026-03-11': {'type': '10Y', 'bid_to_cover': 2.42, 'tail_bps': 0.8},
    '2026-03-12': {'type': '30Y', 'bid_to_cover': 2.22, 'tail_bps': 1.2},
    '2026-04-08': {'type': '10Y', 'bid_to_cover': 2.50, 'tail_bps': 0.3},
    '2026-04-09': {'type': '30Y', 'bid_to_cover': 2.33, 'tail_bps': 0.4},
}

# Edge table
EDGE_TABLE = {
    ('MARKUP', 'CRISIS', 'GOOD_DEMAND'):    {'dir': 'LONG',  'avg': 2.241, 'wr': 0.82, 'n': 11, 'ics_adj': 0.06, 'size_mult': 1.05},
    ('MARKUP', 'NEUTRAL', 'NORMAL'):        {'dir': 'LONG',  'avg': 1.905, 'wr': 0.67, 'n': 9, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('MARKUP', 'TREND', 'NORMAL'):          {'dir': 'LONG',  'avg': 3.025, 'wr': 0.67, 'n': 6, 'ics_adj': 0.06, 'size_mult': 1.05},
    ('DISTRIBUTION', 'TREND', 'GOOD_DEMAND'): {'dir': 'LONG', 'avg': 0.655, 'wr': 0.80, 'n': 5, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('MARKDOWN', 'NEUTRAL', 'GOOD_DEMAND'): {'dir': 'LONG',  'avg': 1.493, 'wr': 0.40, 'n': 10, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('MARKDOWN', 'CRISIS', 'NORMAL'):       {'dir': 'LONG',  'avg': 1.333, 'wr': 0.50, 'n': 6, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('MARKUP', 'NEUTRAL', 'GOOD_DEMAND'):   {'dir': 'SHORT', 'avg': -1.361, 'wr': 0.31, 'n': 16, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('MARKDOWN', 'TREND', 'GOOD_DEMAND'):   {'dir': 'SHORT', 'avg': -1.226, 'wr': 0.27, 'n': 11, 'ics_adj': 0.05, 'size_mult': 1.00},
    ('MARKDOWN', 'NEUTRAL', 'STRONG_DEMAND'): {'dir': 'SHORT', 'avg': -2.900, 'wr': 0.33, 'n': 6, 'ics_adj': 0.06, 'size_mult': 1.05},
    ('MARKDOWN', 'CRISIS', 'STRONG_DEMAND'): {'dir': 'SHORT', 'avg': -3.834, 'wr': 0.25, 'n': 4, 'ics_adj': 0.06, 'size_mult': 1.05},
    ('MARKDOWN', 'TREND', 'WEAK_DEMAND'):   {'dir': 'SHORT', 'avg': -4.591, 'wr': 0.50, 'n': 4, 'ics_adj': 0.06, 'size_mult': 1.05},
}

WEAK_FALLBACK = {'dir': 'SHORT', 'avg': -3.479, 'wr': 0.429, 'n': 14, 'ics_adj': 0.06, 'size_mult': 1.05}

_FRESH_DATA = {}


def update_fresh_data(bid_to_cover, tail_bps, auction_type='10Y'):
    _FRESH_DATA['bid_to_cover'] = bid_to_cover
    _FRESH_DATA['tail_bps'] = tail_bps
    _FRESH_DATA['type'] = auction_type


def _classify_demand(btc, tail, atype):
    if btc is None: return 'NO_DATA'
    if atype == '30Y':
        btc_t = {'STRONG': 2.35, 'GOOD': 2.25, 'WEAK': 2.15}
        tail_t = {'STRONG': 0.3, 'GOOD': 0.7, 'WEAK': 1.2}
    else:
        btc_t = {'STRONG': 2.55, 'GOOD': 2.45, 'WEAK': 2.35}
        tail_t = {'STRONG': 0.3, 'GOOD': 0.6, 'WEAK': 1.0}
    if btc >= btc_t['STRONG'] and (tail or 0) <= tail_t['STRONG']: return 'STRONG_DEMAND'
    elif btc >= btc_t['GOOD'] and (tail or 0) <= tail_t['GOOD']: return 'GOOD_DEMAND'
    elif btc < btc_t['WEAK'] or (tail or 0) > tail_t['WEAK']: return 'WEAK_DEMAND'
    return 'NORMAL'


def _get_release_data(date_str=None):
    if date_str is None: date_str = datetime.utcnow().strftime('%Y-%m-%d')
    if _FRESH_DATA.get('bid_to_cover') is not None: return _FRESH_DATA
    return RECENT_AUCTIONS.get(date_str)


def score_m55_treasury(wyckoff_phase='UNKNOWN', vol_regime='UNKNOWN',
                        direction='LONG', date_str=None):
    release_data = _get_release_data(date_str)
    if release_data is None: return 'NOT_AUCTION_DAY', 0.0, 1.0, {}

    btc = release_data.get('bid_to_cover')
    tail = release_data.get('tail_bps')
    atype = release_data.get('type', '10Y')
    if btc is None: return 'NOT_AUCTION_DAY', 0.0, 1.0, {}

    demand = _classify_demand(btc, tail, atype)
    key = (wyckoff_phase, vol_regime, demand)
    edge = EDGE_TABLE.get(key)
    if edge is None and demand == 'WEAK_DEMAND':
        edge = WEAK_FALLBACK

    if edge is None:
        return 'NO_EDGE', 0.0, 1.0, {
            'demand': demand, 'type': atype, 'bid_to_cover': btc, 'tail_bps': tail,
        }

    if edge['dir'] != direction:
        score_adj, size_mult = -abs(edge['ics_adj']) * 0.5, 0.85
    else:
        score_adj, size_mult = edge['ics_adj'], edge['size_mult']

    details = {
        'demand': demand, 'type': atype, 'bid_to_cover': btc, 'tail_bps': tail,
        'edge_key': f"{wyckoff_phase} × {vol_regime} × {demand}",
        'edge_dir': edge['dir'], 'edge_avg': edge['avg'],
        'edge_wr': edge['wr'], 'edge_n': edge['n'],
    }
    return 'ACTIVE', score_adj, size_mult, details


def format_m55(status, score_adj, size_mult, details):
    if status == 'NOT_AUCTION_DAY': return "M55: — (not Treasury auction day)"
    if status == 'NO_EDGE':
        return f"M55: {details.get('demand','?')} ({details.get('type','?')}, BTC {details.get('bid_to_cover','?')}) — no edge"
    demand = details.get('demand', '?')
    atype = details.get('type', '?')
    btc = details.get('bid_to_cover', '?')
    tail = details.get('tail_bps', '?')
    edge_dir = details.get('edge_dir', '?')
    edge_avg = details.get('edge_avg', 0)
    wr = details.get('edge_wr', 0)
    return (f"M55: {demand} ({atype}, BTC {btc}, tail {tail}bps) | "
            f"{edge_dir} {edge_avg:+.2f}% ({wr:.0%} WR) | "
            f"ICS {score_adj:+.03f} ×{size_mult:.2f}")


if __name__ == '__main__':
    print("=== M55 Treasury Auction Self-Test ===\n")
    for wyck, vol, dire, date in [
        ('MARKUP', 'CRISIS', 'LONG', '2026-04-08'),     # GOOD 10Y
        ('MARKUP', 'NEUTRAL', 'SHORT', '2026-02-11'),    # GOOD 10Y
        ('MARKDOWN', 'TREND', 'SHORT', '2026-03-12'),    # WEAK 30Y
    ]:
        s, a, m, d = score_m55_treasury(wyck, vol, dire, date)
        print(format_m55(s, a, m, d))
        print(f"  → {s}, ICS={a:+.03f}, size={m:.2f}\n")
    print(format_m55(*score_m55_treasury('RANGE', 'NEUTRAL', 'LONG', '2026-01-15')))
