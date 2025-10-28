# Match Data Fetching Guide

## Quick Start

The fetch-and-merge script automatically fetches tennis matches from RapidAPI and merges them into `match_info.csv`.

### Fetch Last Week's Matches (Recommended)

```bash
cd /Users/tudorghica/Desktop/lucky/match-prediction-system
python3 scripts/fetch/fetch_and_merge.py --last-week
```

This will:
- Fetch ATP matches from the past 7 days
- Normalize player names to "Last F." format
- Remove duplicates
- Merge into `data/processed/match_info.csv`
- Save audit file in `data/processed/new_matches_atp_YYYY-MM-DD_YYYY-MM-DD.csv`

## Usage Examples

### 1. Dry Run (Test Without Writing)
```bash
python3 scripts/fetch/fetch_and_merge.py --last-week --dry-run
```

### 2. Specific Date Range
```bash
python3 scripts/fetch/fetch_and_merge.py --start 2025-10-20 --end 2025-10-27
```

### 3. Single Day
```bash
python3 scripts/fetch/fetch_and_merge.py --date 2025-10-27
```

### 4. Fetch Both ATP and WTA
```bash
python3 scripts/fetch/fetch_and_merge.py --last-week --tour atp --tour wta
```

## API Usage

- **Rate Limit**: 500 requests/month
- **Cost per weekly update**: ~20-25 requests
- **Monthly capacity**: ~20 weekly updates (5 months of weekly data!)
- **Recommended schedule**: Weekly updates when you need them (manual)

### Request Breakdown
- 1 request: Get fixtures for date range
- ~10-15 requests: Get tournament names (cached after first fetch)
- ~10-15 requests: Get match results per tournament

**First update**: ~31 requests (no cache)  
**Subsequent updates**: ~16 requests (with tournament name cache)

## What Gets Saved

Each fetch creates two files:

1. **Audit file**: `data/processed/new_matches_atp_YYYY-MM-DD_YYYY-MM-DD.csv`
   - Raw fetched matches (before de-dup)
   - Useful for debugging

2. **Updated match_info.csv**: `data/processed/match_info.csv`
   - Merged dataset with duplicates removed
   - Sorted by date
   - Core columns: date, player1, player2, winner, stadium, surface, description, nation

## Data Quality

✅ **What's included**:
- Player names normalized to "Last F." format
- Surface type (from tournament mapping)
- Tournament name
- Player nations
- Match date
- Winner

✅ **Deduplication**:
- Matches are de-duplicated using: date + tournament + sorted player pair
- Same match won't be added twice

## Troubleshooting

### No matches found
- Date range might be too far in the past (API shows recent matches)
- Try a more recent date range

### Rate limit exceeded
- You have 500 requests/month
- Check how many you've used at: https://rapidapi.com/dashboard
- Weekly updates use ~20-25 requests

### API key error
- API key is hardcoded in `src/data_fetcher/fetch_rapidapi.py`
- Check line with: `self.api_key = api_key or "97c34a0787msh8568dbfa4c6b20fp1cdcabjsn91ce02fe6134"`

## Monthly Usage Tracking

To stay within your 500 request limit:

**Weekly updates** (recommended):
- Week 1: ~25 requests
- Week 2: ~16 requests (cached)
- Week 3: ~16 requests (cached)
- Week 4: ~16 requests (cached)
- **Total**: ~73 requests/month (15% of quota)

**Bi-weekly updates** (very conservative):
- Update 1: ~25 requests
- Update 2: ~16 requests (cached)
- **Total**: ~41 requests/month (8% of quota)

## What's Next?

After fetching new matches:

1. **Regenerate features** if you added many new matches:
   ```bash
   python3 regenerate_features.py
   ```

2. **Retrain model** with updated data:
   ```bash
   python3 train_quick.py
   ```

3. **Check data quality**:
   ```bash
   tail -20 data/processed/match_info.csv
   ```

---

**API Details**: [RapidAPI Tennis API](https://rapidapi.com/jjrm365-kIFr3Nx_odV/api/tennis-api-atp-wta-itf)
