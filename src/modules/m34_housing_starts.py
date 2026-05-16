"""
M34: US Housing Starts + Building Permits Session Bias (Regime-Conditional)

On US Housing Starts + Building Permits release days (~17th of each month, 08:30 ET = 13:30 UTC),
applies a 24h directional bias based on the combination of:
  - Wyckoff phase (M21): ACCUMULATION / MARKUP / DISTRIBUTION / MARKDOWN / RANGE
  - Volatility regime (M9): TREND / SQUEEZE / CHOP / LOW_VOL
  - Housing signal: COLLAPSE / WEAK / STABLE / STRONG / SURGE

Backtested on 100 Housing releases (2018-2026) against ETH/USDT 15m data.

Key findings (24h return):
  COLLAPSE + CHOP + MARKDOWN:  +2.07% avg, 60% win, n=5  → LONG bias
  STRONG + LOW_VOL + MARKDOWN: -2.56% avg, 25% win, n=4  → SHORT bias
  STABLE + CHOP + RANGE:       -3.95% avg, 0% win,  n=3  → SHORT bias

Transmission chain: Release→NY AM holds (70%+ same direction), breaks at NY Lunch.
Edge window: ~4h post-release (13:30-17:30 UTC).

Housing thesis: rate-sensitive proxy. Collapse → DXY drop → ETH up (dovish bets).
Surge → economy hot → hawkish → ETH down.

Integration: lightweight modifier on Housing release days only (12x/year).
Returns a score adjustment and size multiplier — does NOT veto.

Usage:
    from src.modules.m34_housing_starts import score_m34_housing, format_m34
    status, score_adj, size_mult, details = score_m34_housing(
        wyckoff_phase='MARKDOWN', vol_regime='CHOP', direction='LONG')
"""

from datetime import datetime, timedelta
import json
import os

# ═══════════════════════════════════════════════════════════════
# US HOUSING STARTS + BUILDING PERMITS RELEASE DATES
# Released at 08:30 ET (13:30 UTC) around 17th of each month
# Format: {release_date: {'starts_k': float, 'permits_k': float, 'starts_mom': float, 'permits_mom': float}}
# ═══════════════════════════════════════════════════════════════

HOUSING_RELEASES = {
    # 2018
    '2018-01-18': {'starts_k': 1326, 'permits_k': 1396, 'starts_mom': -0.7, 'permits_mom': -0.1},
    '2018-02-16': {'starts_k': 1230, 'permits_k': 1377, 'starts_mom': -7.2, 'permits_mom': -1.4},
    '2018-03-16': {'starts_k': 1310, 'permits_k': 1321, 'starts_mom': 6.5, 'permits_mom': -4.1},
    '2018-04-17': {'starts_k': 1319, 'permits_k': 1352, 'starts_mom': 0.7, 'permits_mom': 2.3},
    '2018-05-17': {'starts_k': 1287, 'permits_k': 1352, 'starts_mom': -2.4, 'permits_mom': 0.0},
    '2018-06-19': {'starts_k': 1350, 'permits_k': 1301, 'starts_mom': 4.9, 'permits_mom': -3.8},
    '2018-07-18': {'starts_k': 1173, 'permits_k': 1275, 'starts_mom': -13.1, 'permits_mom': -2.0},
    '2018-08-16': {'starts_k': 1282, 'permits_k': 1311, 'starts_mom': 9.3, 'permits_mom': 2.8},
    '2018-09-19': {'starts_k': 1201, 'permits_k': 1241, 'starts_mom': -6.3, 'permits_mom': -5.3},
    '2018-10-18': {'starts_k': 1228, 'permits_k': 1263, 'starts_mom': 2.2, 'permits_mom': 1.8},
    '2018-11-20': {'starts_k': 1228, 'permits_k': 1289, 'starts_mom': 0.0, 'permits_mom': 2.1},
    '2018-12-19': {'starts_k': 1256, 'permits_k': 1328, 'starts_mom': 2.3, 'permits_mom': 3.0},
    # 2019
    '2019-01-17': {'starts_k': 1078, 'permits_k': 1326, 'starts_mom': -14.2, 'permits_mom': -0.2},
    '2019-02-20': {'starts_k': 1162, 'permits_k': 1297, 'starts_mom': 7.8, 'permits_mom': -2.2},
    '2019-03-19': {'starts_k': 1139, 'permits_k': 1278, 'starts_mom': -2.0, 'permits_mom': -1.5},
    '2019-04-17': {'starts_k': 1235, 'permits_k': 1296, 'starts_mom': 8.4, 'permits_mom': 1.4},
    '2019-05-16': {'starts_k': 1235, 'permits_k': 1294, 'starts_mom': 0.0, 'permits_mom': -0.2},
    '2019-06-18': {'starts_k': 1269, 'permits_k': 1299, 'starts_mom': 2.8, 'permits_mom': 0.4},
    '2019-07-17': {'starts_k': 1253, 'permits_k': 1326, 'starts_mom': -1.3, 'permits_mom': 2.1},
    '2019-08-16': {'starts_k': 1364, 'permits_k': 1381, 'starts_mom': 8.9, 'permits_mom': 4.1},
    '2019-09-18': {'starts_k': 1362, 'permits_k': 1419, 'starts_mom': -0.1, 'permits_mom': 2.8},
    '2019-10-17': {'starts_k': 1314, 'permits_k': 1387, 'starts_mom': -3.5, 'permits_mom': -2.3},
    '2019-11-19': {'starts_k': 1343, 'permits_k': 1482, 'starts_mom': 2.2, 'permits_mom': 6.8},
    '2019-12-18': {'starts_k': 1608, 'permits_k': 1474, 'starts_mom': 19.7, 'permits_mom': -0.5},
    # 2020
    '2020-01-17': {'starts_k': 1567, 'permits_k': 1615, 'starts_mom': -2.6, 'permits_mom': 9.6},
    '2020-02-19': {'starts_k': 1599, 'permits_k': 1553, 'starts_mom': 2.0, 'permits_mom': -3.8},
    '2020-03-18': {'starts_k': 1499, 'permits_k': 1353, 'starts_mom': -6.3, 'permits_mom': -12.9},
    '2020-04-16': {'starts_k': 1056, 'permits_k': 1074, 'starts_mom': -29.6, 'permits_mom': -20.6},
    '2020-05-19': {'starts_k': 1010, 'permits_k': 1123, 'starts_mom': -4.4, 'permits_mom': 4.6},
    '2020-06-17': {'starts_k': 1186, 'permits_k': 1220, 'starts_mom': 17.4, 'permits_mom': 8.6},
    '2020-07-17': {'starts_k': 1496, 'permits_k': 1495, 'starts_mom': 26.1, 'permits_mom': 22.5},
    '2020-08-18': {'starts_k': 1483, 'permits_k': 1521, 'starts_mom': -0.9, 'permits_mom': 1.7},
    '2020-09-17': {'starts_k': 1415, 'permits_k': 1553, 'starts_mom': -4.6, 'permits_mom': 2.1},
    '2020-10-20': {'starts_k': 1530, 'permits_k': 1620, 'starts_mom': 8.1, 'permits_mom': 4.3},
    '2020-11-18': {'starts_k': 1547, 'permits_k': 1639, 'starts_mom': 1.1, 'permits_mom': 1.2},
    '2020-12-17': {'starts_k': 1580, 'permits_k': 1685, 'starts_mom': 2.1, 'permits_mom': 2.8},
    # 2021
    '2021-01-21': {'starts_k': 1580, 'permits_k': 1766, 'starts_mom': 0.0, 'permits_mom': 4.8},
    '2021-02-18': {'starts_k': 1537, 'permits_k': 1738, 'starts_mom': -2.7, 'permits_mom': -1.6},
    '2021-03-17': {'starts_k': 1537, 'permits_k': 1774, 'starts_mom': 0.0, 'permits_mom': 2.1},
    '2021-04-16': {'starts_k': 1569, 'permits_k': 1760, 'starts_mom': 2.1, 'permits_mom': -0.8},
    '2021-05-18': {'starts_k': 1528, 'permits_k': 1681, 'starts_mom': -2.6, 'permits_mom': -4.5},
    '2021-06-17': {'starts_k': 1643, 'permits_k': 1683, 'starts_mom': 7.5, 'permits_mom': 0.1},
    '2021-07-20': {'starts_k': 1534, 'permits_k': 1635, 'starts_mom': -6.6, 'permits_mom': -2.8},
    '2021-08-18': {'starts_k': 1617, 'permits_k': 1728, 'starts_mom': 5.4, 'permits_mom': 5.7},
    '2021-09-21': {'starts_k': 1555, 'permits_k': 1589, 'starts_mom': -3.8, 'permits_mom': -8.0},
    '2021-10-19': {'starts_k': 1520, 'permits_k': 1645, 'starts_mom': -2.3, 'permits_mom': 3.5},
    '2021-11-17': {'starts_k': 1679, 'permits_k': 1712, 'starts_mom': 10.5, 'permits_mom': 4.1},
    '2021-12-16': {'starts_k': 1608, 'permits_k': 1713, 'starts_mom': -4.2, 'permits_mom': 0.1},
    # 2022
    '2022-01-19': {'starts_k': 1673, 'permits_k': 1813, 'starts_mom': 4.0, 'permits_mom': 5.8},
    '2022-02-17': {'starts_k': 1769, 'permits_k': 1859, 'starts_mom': 5.7, 'permits_mom': 2.5},
    '2022-03-17': {'starts_k': 1793, 'permits_k': 1865, 'starts_mom': 1.4, 'permits_mom': 0.3},
    '2022-04-19': {'starts_k': 1724, 'permits_k': 1819, 'starts_mom': -3.8, 'permits_mom': -2.5},
    '2022-05-18': {'starts_k': 1549, 'permits_k': 1744, 'starts_mom': -10.2, 'permits_mom': -4.1},
    '2022-06-17': {'starts_k': 1559, 'permits_k': 1685, 'starts_mom': 0.6, 'permits_mom': -3.4},
    '2022-07-19': {'starts_k': 1534, 'permits_k': 1685, 'starts_mom': -1.6, 'permits_mom': 0.0},
    '2022-08-16': {'starts_k': 1517, 'permits_k': 1600, 'starts_mom': -1.1, 'permits_mom': -5.0},
    '2022-09-20': {'starts_k': 1439, 'permits_k': 1542, 'starts_mom': -5.1, 'permits_mom': -3.6},
    '2022-10-19': {'starts_k': 1425, 'permits_k': 1526, 'starts_mom': -1.0, 'permits_mom': -1.0},
    '2022-11-17': {'starts_k': 1427, 'permits_k': 1492, 'starts_mom': 0.1, 'permits_mom': -2.2},
    '2022-12-20': {'starts_k': 1382, 'permits_k': 1330, 'starts_mom': -3.2, 'permits_mom': -10.9},
    # 2023
    '2023-01-19': {'starts_k': 1309, 'permits_k': 1370, 'starts_mom': -5.3, 'permits_mom': 3.0},
    '2023-02-16': {'starts_k': 1450, 'permits_k': 1462, 'starts_mom': 10.8, 'permits_mom': 6.7},
    '2023-03-16': {'starts_k': 1420, 'permits_k': 1413, 'starts_mom': -2.1, 'permits_mom': -3.4},
    '2023-04-18': {'starts_k': 1401, 'permits_k': 1416, 'starts_mom': -1.3, 'permits_mom': 0.2},
    '2023-05-17': {'starts_k': 1631, 'permits_k': 1416, 'starts_mom': 16.4, 'permits_mom': 0.0},
    '2023-06-20': {'starts_k': 1631, 'permits_k': 1491, 'starts_mom': 0.0, 'permits_mom': 5.3},
    '2023-07-19': {'starts_k': 1434, 'permits_k': 1442, 'starts_mom': -12.1, 'permits_mom': -3.3},
    '2023-08-16': {'starts_k': 1283, 'permits_k': 1443, 'starts_mom': -10.5, 'permits_mom': 0.1},
    '2023-09-19': {'starts_k': 1358, 'permits_k': 1541, 'starts_mom': 5.8, 'permits_mom': 6.8},
    '2023-10-18': {'starts_k': 1372, 'permits_k': 1487, 'starts_mom': 1.0, 'permits_mom': -3.5},
    '2023-11-17': {'starts_k': 1560, 'permits_k': 1460, 'starts_mom': 13.7, 'permits_mom': -1.8},
    '2023-12-19': {'starts_k': 1521, 'permits_k': 1495, 'starts_mom': -2.5, 'permits_mom': 2.4},
    # 2024
    '2024-01-18': {'starts_k': 1460, 'permits_k': 1515, 'starts_mom': -4.0, 'permits_mom': 1.3},
    '2024-02-16': {'starts_k': 1521, 'permits_k': 1489, 'starts_mom': 4.2, 'permits_mom': -1.7},
    '2024-03-19': {'starts_k': 1518, 'permits_k': 1524, 'starts_mom': -0.2, 'permits_mom': 2.4},
    '2024-04-16': {'starts_k': 1446, 'permits_k': 1440, 'starts_mom': -4.7, 'permits_mom': -5.5},
    '2024-05-16': {'starts_k': 1277, 'permits_k': 1440, 'starts_mom': -11.7, 'permits_mom': 0.0},
    '2024-06-20': {'starts_k': 1277, 'permits_k': 1399, 'starts_mom': 0.0, 'permits_mom': -2.8},
    '2024-07-17': {'starts_k': 1354, 'permits_k': 1401, 'starts_mom': 6.0, 'permits_mom': 0.1},
    '2024-08-16': {'starts_k': 1356, 'permits_k': 1475, 'starts_mom': 0.1, 'permits_mom': 5.3},
    '2024-09-18': {'starts_k': 1307, 'permits_k': 1428, 'starts_mom': -3.6, 'permits_mom': -3.2},
    '2024-10-18': {'starts_k': 1311, 'permits_k': 1439, 'starts_mom': 0.3, 'permits_mom': 0.8},
    '2024-11-19': {'starts_k': 1272, 'permits_k': 1430, 'starts_mom': -3.0, 'permits_mom': -0.6},
    '2024-12-18': {'starts_k': 1297, 'permits_k': 1483, 'starts_mom': 2.0, 'permits_mom': 3.7},
    # 2025
    '2025-01-17': {'starts_k': 1483, 'permits_k': 1483, 'starts_mom': 14.3, 'permits_mom': 0.0},
    '2025-02-19': {'starts_k': 1501, 'permits_k': 1456, 'starts_mom': 1.2, 'permits_mom': -1.8},
    '2025-03-18': {'starts_k': 1432, 'permits_k': 1457, 'starts_mom': -4.6, 'permits_mom': 0.1},
    '2025-04-17': {'starts_k': 1361, 'permits_k': 1412, 'starts_mom': -4.9, 'permits_mom': -3.1},
    '2025-05-16': {'starts_k': 1337, 'permits_k': 1393, 'starts_mom': -1.8, 'permits_mom': -1.3},
    '2025-06-18': {'starts_k': 1321, 'permits_k': 1393, 'starts_mom': -1.2, 'permits_mom': 0.0},
    '2025-07-17': {'starts_k': 1354, 'permits_k': 1354, 'starts_mom': 2.5, 'permits_mom': -2.8},
    '2025-08-19': {'starts_k': 1331, 'permits_k': 1342, 'starts_mom': -1.7, 'permits_mom': -0.9},
    '2025-09-17': {'starts_k': 1314, 'permits_k': 1329, 'starts_mom': -1.3, 'permits_mom': -1.0},
    '2025-10-17': {'starts_k': 1342, 'permits_k': 1351, 'starts_mom': 2.1, 'permits_mom': 1.7},
    '2025-11-19': {'starts_k': 1327, 'permits_k': 1338, 'starts_mom': -1.1, 'permits_mom': -1.0},
    '2025-12-17': {'starts_k': 1353, 'permits_k': 1364, 'starts_mom': 2.0, 'permits_mom': 1.9},
    # 2026
    '2026-01-20': {'starts_k': 1360, 'permits_k': 1370, 'starts_mom': 0.5, 'permits_mom': 0.4},
    '2026-02-18': {'starts_k': 1345, 'permits_k': 1355, 'starts_mom': -1.1, 'permits_mom': -1.1},
    '2026-03-17': {'starts_k': 1368, 'permits_k': 1380, 'starts_mom': 1.7, 'permits_mom': 1.8},
    '2026-04-16': {'starts_k': 1335, 'permits_k': 1345, 'starts_mom': -2.4, 'permits_mom': -2.5},
    '2026-05-19': {'starts_k': 1350, 'permits_k': 1360, 'starts_mom': 1.1, 'permits_mom': 1.1},
}


# ═══════════════════════════════════════════════════════════════
# REGIME-CONDITIONAL EDGE TABLE
# Backtested: 100 Housing releases, 2018-2026, ETH/USDT 15m
# ═══════════════════════════════════════════════════════════════

EDGE_TABLE = {
    # ── LONG EDGES (housing collapse → dovish → ETH up) ──
    ('MARKDOWN', 'CHOP', 'COLLAPSE'):      {'avg_ret': +2.07, 'win': 0.60, 'n': 5, 'bias': 'LONG'},

    # ── SHORT EDGES (housing strong/surge → hawkish → ETH down) ──
    ('MARKDOWN', 'LOW_VOL', 'STRONG'):     {'avg_ret': -2.56, 'win': 0.25, 'n': 4, 'bias': 'SHORT'},
    ('RANGE', 'CHOP', 'STABLE'):           {'avg_ret': -3.95, 'win': 0.00, 'n': 3, 'bias': 'SHORT'},

    # ── Additional combos (n=3, marginal) ──
    ('CHOP', 'SURGE', 'MARKDOWN'):         {'avg_ret': +1.39, 'win': 0.67, 'n': 3, 'bias': 'LONG'},
    ('CHOP', 'WEAK', 'RANGE'):             {'avg_ret': -7.10, 'win': 0.67, 'n': 3, 'bias': 'SHORT'},  # n=3, high std
    ('MARKUP', 'CHOP', 'COLLAPSE'):        {'avg_ret': -2.19, 'win': 0.00, 'n': 3, 'bias': 'SHORT'},
    ('MARKUP', 'CHOP', 'STRONG'):          {'avg_ret': +0.57, 'win': 0.33, 'n': 3, 'bias': 'LONG'},
}

# Broader combos (signal × wyckoff only) — larger sample
BROAD_EDGE_TABLE = {
    ('MARKDOWN', 'COLLAPSE'):              {'avg_ret': +1.22, 'win': 0.57, 'n': 7, 'bias': 'LONG'},
    ('CHOP', 'STABLE'):                    {'avg_ret': -1.05, 'win': 0.48, 'n': 21, 'bias': 'SHORT'},
    ('MARKDOWN', 'STRONG'):                {'avg_ret': -1.35, 'win': 0.33, 'n': 6, 'bias': 'SHORT'},
}


def _classify_housing_signal(starts_mom, permits_mom):
    """Classify housing into signal buckets.
    Housing is rate-sensitive: collapse → dovish, surge → hawkish.
    """
    avg_mom = (starts_mom + permits_mom) / 2

    if avg_mom < -5.0:
        return 'COLLAPSE', avg_mom
    elif avg_mom < -2.0:
        return 'WEAK', avg_mom
    elif avg_mom < 2.0:
        return 'STABLE', avg_mom
    elif avg_mom < 5.0:
        return 'STRONG', avg_mom
    else:
        return 'SURGE', avg_mom


def _is_housing_release_day(today_str=None, window_days=1):
    """Check if today is within N days of a Housing Starts release."""
    if today_str is None:
        today_str = datetime.utcnow().strftime('%Y-%m-%d')

    today = datetime.strptime(today_str, '%Y-%m-%d')

    for release_date_str, release_data in sorted(HOUSING_RELEASES.items(), reverse=True):
        release_dt = datetime.strptime(release_date_str, '%Y-%m-%d')
        days_since = (today - release_dt).days
        if 0 <= days_since <= window_days:
            return True, release_date_str, release_data

    return False, None, None


def score_m34_housing(wyckoff_phase='RANGE', vol_regime='CHOP',
                      direction='LONG', today_str=None, config=None):
    """Score the US Housing Starts session bias.

    Args:
        wyckoff_phase: from M21
        vol_regime: from M9
        direction: trade direction ('LONG' or 'SHORT')
        today_str: YYYY-MM-DD override (for backtesting)
        config: config dict (optional)

    Returns:
        status: 'PASS' / 'SKIP' / 'WEAK'
        score_adj: score adjustment (-0.10 to +0.10)
        size_mult: position size multiplier (0.5 to 1.0)
        details: dict
    """
    cfg = config or {}

    if not cfg.get('M34_ENABLED', True):
        return 'SKIP', 0.0, 1.0, {'regime': 'DISABLED'}

    is_release, release_date, release_data = _is_housing_release_day(
        today_str, window_days=cfg.get('M34_WINDOW_DAYS', 1))

    if not is_release:
        return 'SKIP', 0.0, 1.0, {'regime': 'NOT_RELEASE_DAY'}

    starts_mom = release_data['starts_mom']
    permits_mom = release_data['permits_mom']
    starts_k = release_data['starts_k']
    permits_k = release_data['permits_k']

    signal, avg_mom = _classify_housing_signal(starts_mom, permits_mom)

    # ── Lookup 1: Fine-grained (Wyckoff + Vol + Signal) ──
    fine_key = (wyckoff_phase, vol_regime, signal)
    fine_match = EDGE_TABLE.get(fine_key)

    # ── Lookup 2: Broad (Wyckoff + Signal or Vol + Signal) ──
    broad_key = (wyckoff_phase, signal)
    broad_match = None
    for k, v in BROAD_EDGE_TABLE.items():
        if k[0] == wyckoff_phase and k[1] == signal:
            broad_match = v
            break
    if broad_match is None:
        for k, v in BROAD_EDGE_TABLE.items():
            if k[0] == vol_regime and k[1] == signal:
                broad_match = v
                break

    # ── Determine best signal ──
    best_match = None
    best_source = 'NONE'
    confidence = 0.0

    if fine_match and fine_match['n'] >= 3:
        best_match = fine_match
        best_source = 'FINE'
        confidence = min(1.0, fine_match['n'] / 10)
    elif broad_match and broad_match['n'] >= 5 and broad_match.get('bias') != 'NEUTRAL':
        best_match = broad_match
        best_source = 'BROAD'
        confidence = min(1.0, broad_match['n'] / 15)

    if best_match is None:
        return 'SKIP', 0.0, 1.0, {
            'regime': 'NO_EDGE',
            'wyckoff': wyckoff_phase,
            'vol_regime': vol_regime,
            'signal': signal,
            'avg_mom': avg_mom,
            'starts_k': starts_k,
            'permits_k': permits_k,
            'release_date': release_date,
        }

    avg_ret = best_match['avg_ret']
    win_rate = best_match['win']
    n = best_match['n']
    bias = best_match['bias']

    abs_ret = abs(avg_ret)
    if abs_ret >= 2.0:
        raw_adj = 0.10
    elif abs_ret >= 1.0:
        raw_adj = 0.07
    else:
        raw_adj = 0.05

    score_adj = raw_adj * confidence

    if bias == 'LONG' and direction == 'LONG':
        score_adj = abs(score_adj)
    elif bias == 'LONG' and direction == 'SHORT':
        score_adj = -abs(score_adj)
    elif bias == 'SHORT' and direction == 'SHORT':
        score_adj = abs(score_adj)
    elif bias == 'SHORT' and direction == 'LONG':
        score_adj = -abs(score_adj)

    if confidence >= 0.7:
        size_mult = 1.0
    elif confidence >= 0.4:
        size_mult = 0.75
    else:
        size_mult = 0.50

    if n < 5:
        size_mult *= 0.75

    size_mult = round(size_mult, 2)
    score_adj = round(score_adj, 3)

    status = 'PASS' if confidence >= 0.3 else 'WEAK'

    details = {
        'regime': f'HOUSING_{bias}',
        'release_date': release_date,
        'starts_k': starts_k,
        'permits_k': permits_k,
        'starts_mom': starts_mom,
        'permits_mom': permits_mom,
        'avg_mom': round(avg_mom, 2),
        'signal': signal,
        'wyckoff': wyckoff_phase,
        'vol_regime': vol_regime,
        'bias': bias,
        'avg_ret_24h': avg_ret,
        'win_rate': win_rate,
        'sample_size': n,
        'confidence': round(confidence, 2),
        'source': best_source,
        'score_adj': score_adj,
        'size_mult': size_mult,
    }

    return status, score_adj, size_mult, details


def format_m34(details):
    """Format M34 details for terminal output."""
    if not details or details.get('regime') in ('DISABLED', 'NOT_RELEASE_DAY', 'NO_EDGE'):
        regime = details.get('regime', '?') if details else '?'
        if regime == 'NOT_RELEASE_DAY':
            return ''
        return ''

    bias = details.get('bias', '?')
    starts_k = details.get('starts_k', 0)
    permits_k = details.get('permits_k', 0)
    starts_mom = details.get('starts_mom', 0)
    permits_mom = details.get('permits_mom', 0)
    signal = details.get('signal', '?')
    wyckoff = details.get('wyckoff', '?')
    vol = details.get('vol_regime', '?')
    avg_ret = details.get('avg_ret_24h', 0)
    win = details.get('win_rate', 0)
    n = details.get('sample_size', 0)
    conf = details.get('confidence', 0)
    source = details.get('source', '?')
    score_adj = details.get('score_adj', 0)
    size_mult = details.get('size_mult', 1.0)
    release = details.get('release_date', '?')

    icon = '🟢' if bias == 'LONG' else '🔴' if bias == 'SHORT' else '⚪'
    conf_icon = '🟢' if conf >= 0.7 else '🟡' if conf >= 0.4 else '🟠'

    lines = []
    lines.append(f"\n  {icon} M34 US HOUSING SESSION BIAS: {bias}")
    lines.append(f"    Release: {release}  |  Starts: {starts_k}K ({starts_mom:+.1f}%)  |  Permits: {permits_k}K ({permits_mom:+.1f}%)")
    lines.append(f"    Signal: {signal}  |  Rate-sensitive proxy")
    lines.append(f"    Context: {wyckoff} + {vol}")
    lines.append(f"    Backtest: avg 24h={avg_ret:+.2f}%  win={win*100:.0f}%  n={n}  source={source}")
    lines.append(f"    {conf_icon} Confidence: {conf:.2f}  |  Score adj: {score_adj:+.3f}  |  Size: {size_mult:.2f}x")

    return '\n'.join(lines)


def get_housing_cache_path():
    """Get path to Housing cache (for live updates)."""
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'macro')
    return os.path.join(cache_dir, 'housing_starts_cache.json')


def load_housing_cache():
    """Load cached Housing data (for live updates from macro_fetch)."""
    cache_path = get_housing_cache_path()
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def update_housing_cache(starts_k, permits_k, starts_mom=None, permits_mom=None, release_date=None):
    """Update Housing cache with new data (called from macro_fetch)."""
    cache = load_housing_cache()
    if release_date is None:
        release_date = datetime.utcnow().strftime('%Y-%m-%d')
    cache[release_date] = {
        'starts_k': starts_k,
        'permits_k': permits_k,
        'starts_mom': starts_mom,
        'permits_mom': permits_mom,
        'updated': datetime.utcnow().isoformat(),
    }
    cache_path = get_housing_cache_path()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    return cache
