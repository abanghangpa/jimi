#!/usr/bin/env python3
"""
Update manual PMI data file.

Usage:
    python3 scripts/update_pmi.py --caixin 50.7 --nbs 49.4
    python3 scripts/update_pmi.py --caixin 50.7 --previous 51.2 --nbs 49.4 --nbs-previous 50.5
"""

import argparse
import json
import os
from datetime import datetime, timezone

UTC = timezone.utc


def main():
    parser = argparse.ArgumentParser(description='Update manual PMI data')
    parser.add_argument('--caixin', type=float, required=True, help='Caixin Mfg PMI actual')
    parser.add_argument('--previous', type=float, help='Caixin previous value')
    parser.add_argument('--forecast', type=float, help='Caixin forecast value')
    parser.add_argument('--nbs', type=float, help='NBS Mfg PMI actual')
    parser.add_argument('--nbs-previous', type=float, help='NBS previous value')
    parser.add_argument('--month', type=str, help='Month key (YYYY-MM), default: current month')
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'macro')
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, 'manual_pmi.json')

    # Load existing
    if os.path.exists(filepath):
        with open(filepath) as f:
            data = json.load(f)
    else:
        data = {}

    # Month key
    if args.month:
        month_key = args.month
    else:
        now = datetime.now(UTC)
        month_key = f"{now.year}-{now.month:02d}"

    # Build entry
    entry = {
        'actual': args.caixin,
        'previous': args.previous or data.get(month_key, {}).get('previous', args.caixin),
        'forecast': args.forecast or data.get(month_key, {}).get('forecast'),
    }

    if args.nbs:
        entry['nbs_actual'] = args.nbs
        entry['nbs_previous'] = args.nbs_previous or data.get(month_key, {}).get('nbs_previous', args.nbs)

    data[month_key] = entry

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ Updated {month_key}:")
    print(f"   Caixin: {entry['actual']} (prev: {entry['previous']})")
    if args.nbs:
        print(f"   NBS:    {entry.get('nbs_actual')} (prev: {entry.get('nbs_previous')})")
    print(f"   Saved to: {filepath}")


if __name__ == '__main__':
    main()
