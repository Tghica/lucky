#!/usr/bin/env python3
"""
Fetch tennis matches from RapidAPI Tennis API, normalize to internal schema, and append to
`data/processed/match_info.csv` with de-duplication.

Designed for weekly batch fetching (stays within free tier: 100 requests/month).

Usage examples:
  # Fetch last week's ATP matches
  python scripts/fetch/fetch_and_merge.py --last-week

  # Fetch specific date range
  python scripts/fetch/fetch_and_merge.py --start 2025-10-20 --end 2025-10-27

  # Fetch specific date (YYYY-MM-DD)
  python scripts/fetch/fetch_and_merge.py --date 2025-10-27

  # Fetch both ATP and WTA
  python scripts/fetch/fetch_and_merge.py --last-week --tour atp --tour wta

  # Dry run without writing
  python scripts/fetch/fetch_and_merge.py --last-week --dry-run

Setup:
  API key is configured in src/data_fetcher/fetch_rapidapi.py
  You have 500 requests/month - enough for ~20 weekly updates!

Notes:
- RapidAPI provides surface type and tournament metadata (advantage over ESPN)
- 500 requests/month supports weekly updates with plenty of buffer
- De-duplication uses date + tournament + player pair to prevent duplicates
- Each weekly update uses ~20-25 requests (~73 requests/month for 4 weekly updates)
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Local imports
sys.path.append(str(Path(__file__).resolve().parents[2]))  # add project root to path
from src.data_fetcher import fetch_tennis_matches  # type: ignore


CORE_COLUMNS = [
    'date', 'player1', 'player2', 'winner', 'stadium', 'surface', 'description', 'nation'
]


def normalize_player_name(name: str) -> str:
    """Convert names like 'Roger Federer' -> 'Federer R.'; handle simple multi-part names.
    Assumptions: last token is the family name; use first letter of the first token as initial.
    """
    if not isinstance(name, str) or not name.strip():
        return name
    parts = name.replace('\u00a0', ' ').strip().split()
    if len(parts) == 1:
        # e.g., mononyms
        return parts[0]
    first_initial = parts[0][0].upper()
    last_name = parts[-1]
    return f"{last_name} {first_initial}."


def normalize_match_record(raw: Dict) -> Dict:
    """Map RapidAPI record to internal schema and normalize fields."""
    # Date -> 'YYYY-MM-DD' (already in correct format from API)
    date_str = raw.get('date', '')
    if not date_str:
        date_str = datetime.now().strftime('%Y-%m-%d')

    p1 = normalize_player_name(raw.get('player1'))
    p2 = normalize_player_name(raw.get('player2'))

    # Winner: normalize
    winner_raw = raw.get('winner')
    winner_norm = normalize_player_name(winner_raw) if winner_raw else None
    if winner_norm not in {p1, p2}:
        # Try to match by comparing raw names
        if winner_raw and winner_raw.strip() == raw.get('player1', '').strip():
            winner_norm = p1
        elif winner_raw and winner_raw.strip() == raw.get('player2', '').strip():
            winner_norm = p2
        else:
            winner_norm = None

    mapped = {
        'date': date_str,
        'player1': p1,
        'player2': p2,
        'winner': winner_norm,
        'stadium': raw.get('stadium') or None,
        'surface': raw.get('surface') or None,  # RapidAPI provides this!
        'description': raw.get('description') or None,
        'nation': raw.get('nation') or None,
    }
    return mapped


def fetch_range(start: str, end: str, tours: List[str]) -> List[Dict]:
    """Fetch records for date range from RapidAPI; dates are YYYY-MM-DD strings."""
    all_records: List[Dict] = []
    
    try:
        # Use RapidAPI client to fetch matches for all tours
        records = fetch_tennis_matches(start, end, tours=tours)
        for r in records:
            all_records.append(normalize_match_record(r))
        print(f"  Fetched {len(records)} matches from {', '.join(t.upper() for t in tours)}")
    except Exception as e:
        print(f"  Warning: Fetch failed: {e}")
    
    return all_records


def build_canonical_key(row: pd.Series) -> str:
    """Order-insensitive key for de-dup: date + description + sorted player pair."""
    date = row.get('date') or ''
    desc = row.get('description') or ''
    p1 = row.get('player1') or ''
    p2 = row.get('player2') or ''
    a, b = sorted([p1, p2])
    return f"{date}|{desc}|{a}|{b}"


def ensure_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in CORE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[CORE_COLUMNS]


def main():
    parser = argparse.ArgumentParser(
        description='Fetch RapidAPI tennis matches and merge into match_info.csv',
        epilog='API key configured in src/data_fetcher/fetch_rapidapi.py (500 requests/month)'
    )
    
    # Date options (mutually exclusive)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--date', help='Single date YYYY-MM-DD')
    g.add_argument('--start', help='Start date YYYY-MM-DD (requires --end)')
    g.add_argument('--last-week', action='store_true', help='Fetch last 7 days')
    
    parser.add_argument('--end', help='End date YYYY-MM-DD (required with --start)')
    parser.add_argument('--tour', action='append', choices=['atp', 'wta'], 
                       default=None, help='Tour(s) to fetch (default: atp)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Preview without writing files')

    args = parser.parse_args()

    # Determine date range
    if args.last_week:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        start = start_date.strftime('%Y-%m-%d')
        end = end_date.strftime('%Y-%m-%d')
    elif args.date:
        start = end = args.date
    else:
        if not args.end:
            parser.error('--end is required when using --start')
        start, end = args.start, args.end
    
    # Default to ATP if no tours specified
    tours = args.tour if args.tour else ['atp']

    print(f"Fetching {', '.join(t.upper() for t in tours)} matches: {start} to {end}...")
    
    try:
        new_records = fetch_range(start, end, tours)
    except Exception as e:
        print(f"\n❌ Fetch failed: {e}")
        print("\nTroubleshooting:")
        print("- Check API key in src/data_fetcher/fetch_rapidapi.py")
        print("- Verify you haven't exceeded rate limits (500 req/month)")
        print("- Review API docs: https://rapidapi.com/jjrm365-kIFr3Nx_odV/api/tennis-api-atp-wta-itf")
        sys.exit(1)
    
    new_df = pd.DataFrame(new_records)
    new_df = ensure_core_columns(new_df)

    # Paths
    root = Path(__file__).resolve().parents[2]
    match_info_path = root / 'data' / 'processed' / 'match_info.csv'
    audit_dir = root / 'data' / 'processed'
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    # Create descriptive audit filename
    tours_str = '_'.join(tours)
    if start != end:
        audit_name = f"new_matches_{tours_str}_{start}_{end}.csv"
    else:
        audit_name = f"new_matches_{tours_str}_{start}.csv"
    audit_path = audit_dir / audit_name

    print(f"Fetched: {len(new_df)} matches")
    if len(new_df) == 0:
        print("No matches found.")
        if not args.dry_run:
            new_df.to_csv(audit_path, index=False)
            print(f"Wrote empty audit file to {audit_path}")
        return

    # Load existing match_info if present
    if match_info_path.exists():
        existing_df = pd.read_csv(match_info_path)
        existing_df = ensure_core_columns(existing_df)
    else:
        print("No existing match_info.csv found; will create a new one with core columns only.")
        existing_df = pd.DataFrame(columns=CORE_COLUMNS)

    # De-dup using canonical key
    new_df['__key'] = new_df.apply(build_canonical_key, axis=1)
    existing_df['__key'] = existing_df.apply(build_canonical_key, axis=1)

    before_unique = len(new_df)
    new_df = new_df[~new_df['__key'].isin(existing_df['__key'])]
    removed = before_unique - len(new_df)

    print(f"New unique matches: {len(new_df)} (removed {removed} as potential duplicates)")

    # Write audit
    if not args.dry_run:
        pd.DataFrame(new_records).to_csv(audit_path, index=False)
        print(f"Audit file written to: {audit_path}")

    if len(new_df) == 0:
        print("Nothing to merge; exiting.")
        return

    # Merge and sort
    merged = pd.concat([
        existing_df.drop(columns=['__key'], errors='ignore'),
        new_df.drop(columns=['__key'], errors='ignore')
    ], ignore_index=True)

    # Sort by date if possible
    try:
        merged['date'] = pd.to_datetime(merged['date'])
        merged = merged.sort_values('date').reset_index(drop=True)
        merged['date'] = merged['date'].dt.strftime('%Y-%m-%d')
    except Exception:
        pass

    if args.dry_run:
        print("Dry run: not writing match_info.csv")
        print(merged.tail(5))
    else:
        match_info_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(match_info_path, index=False)
        print(f"Merged dataset written to: {match_info_path}")
        print(f"Total matches now: {len(merged)}")


if __name__ == '__main__':
    main()
