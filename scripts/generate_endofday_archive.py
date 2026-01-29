#!/usr/bin/env python3
"""Fetch Synoptic timeseries for a CST daytime window and save to archive/EndOfDay.

Usage:
  python generate_endofday_archive.py [--date YYYY-MM-DD]

Defaults to yesterday (US/Central) if no date provided.
"""
import os
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pytz

from pathlib import Path
import sys

# Robustly find the project directory (works when running in-container
# where the mounted folder may be the repo root or the contents of `api/`).
FILE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = None
for p in [FILE_DIR] + list(FILE_DIR.parents)[:4]:
    if (p / 'api').is_dir() or (p / 'services').is_dir():
        PROJECT_DIR = str(p)
        break
if PROJECT_DIR is None:
    PROJECT_DIR = str(FILE_DIR.parent)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import asyncio

# Import the synoptic helper. In some container setups the code under
# `/app` is the contents of the `api/` package (i.e. services/ lives at
# PROJECT_DIR/services). To allow `from api.services...` to work in both
# host and container layouts, create a lightweight package shim for
# `api` pointing at PROJECT_DIR if needed.
try:
    from api.services.synoptic import fetch_historical_station_data
except ModuleNotFoundError as e:
    if 'api' in str(e):
        import types
        api_pkg = types.ModuleType('api')
        api_pkg.__path__ = [PROJECT_DIR]
        sys.modules['api'] = api_pkg
        from api.services.synoptic import fetch_historical_station_data
    else:
        raise


def make_window_for_date(date_obj):
    """Return (start_utc, end_utc) for 10:00-21:00 US/Central on date_obj."""
    central = pytz.timezone('US/Central')
    start_local = central.localize(datetime(date_obj.year, date_obj.month, date_obj.day, 10, 0, 0))
    end_local = central.localize(datetime(date_obj.year, date_obj.month, date_obj.day, 21, 0, 0))
    start_utc = start_local.astimezone(pytz.UTC)
    end_utc = end_local.astimezone(pytz.UTC)
    return start_utc, end_utc


def save_response(data, out_dir, date_obj):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f'endofday_raw_{date_obj.strftime("%Y%m%d")}.json'
    path = out_dir / fname
    # add metadata
    data['_end_of_day_for'] = date_obj.strftime('%Y-%m-%d')
    data['_saved_at'] = datetime.utcnow().isoformat()
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    return str(path)


def main(target_date=None, states=None, networks=None):
    # default target_date: yesterday in US/Central
    central = pytz.timezone('US/Central')
    if target_date:
        date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
    else:
        now_central = datetime.now(central)
        date_obj = (now_central - timedelta(days=1)).date()

    if states is None:
        # Include Missouri plus neighboring states to improve coverage
        states = ['MO', 'OK', 'AR', 'TN', 'KY', 'IL', 'IA', 'NE', 'KS']
    if networks is None:
        networks = [1, 2, 165]

    start_utc, end_utc = make_window_for_date(date_obj)

    print(f'Fetching timeseries for {date_obj} ({start_utc.isoformat()} -> {end_utc.isoformat()}) networks={networks} states={states}')

    try:
        data = asyncio.run(fetch_historical_station_data(states=states, start_time=start_utc, end_time=end_utc, networks=networks))
    except RuntimeError:
        # If already in loop
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        data = loop.run_until_complete(fetch_historical_station_data(states=states, start_time=start_utc, end_time=end_utc, networks=networks))

    out_dir = os.path.join(PROJECT_DIR, 'archive', 'EndOfDay')
    saved = save_response(data, out_dir, date_obj)
    print('Saved end-of-day archive to', saved)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', '-d', help='Date YYYY-MM-DD (defaults to yesterday CST)')
    parser.add_argument('--states', '-s', help='Comma-separated states (default: MO + neighboring states)')
    parser.add_argument('--networks', '-n', help='Comma-separated network IDs (default: 1,2,165)')
    args = parser.parse_args()

    states = args.states.split(',') if args.states else None
    networks = list(map(int, args.networks.split(','))) if args.networks else None
    main(target_date=args.date, states=states, networks=networks)
