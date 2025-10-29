#!/usr/bin/env python3
"""
Comprehensive data quality and inconsistency checker.
Analyzes matches.csv, players.csv, and stadiums.csv for issues.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def check_matches_data():
    """Check matches.csv for inconsistencies"""
    print("="*70)
    print("MATCHES.CSV - DATA QUALITY CHECK")
    print("="*70)
    
    matches = pd.read_csv('data/processed/matches.csv', low_memory=False)
    issues = []
    
    print(f"\n📊 Dataset: {len(matches):,} matches")
    print(f"Date range: {matches['date'].min()} to {matches['date'].max()}")
    
    # 1. Check for duplicate matches
    print("\n1️⃣ DUPLICATE MATCHES:")
    matches['date_str'] = pd.to_datetime(matches['date']).dt.strftime('%Y-%m-%d')
    duplicates = matches.duplicated(subset=['date_str', 'winner_name', 'loser_name', 'tournament'], keep=False)
    if duplicates.sum() > 0:
        print(f"   ❌ Found {duplicates.sum()} duplicate matches")
        issues.append(f"Duplicate matches: {duplicates.sum()}")
        dup_sample = matches[duplicates].head(5)[['date', 'tournament', 'winner_name', 'loser_name']]
        print(dup_sample.to_string(index=False))
    else:
        print("   ✅ No duplicates found")
    
    # 2. Check winner vs loser consistency
    print("\n2️⃣ WINNER/LOSER LOGIC:")
    # Check if winner and loser are same person
    same_player = matches['winner_name'] == matches['loser_name']
    if same_player.sum() > 0:
        print(f"   ❌ {same_player.sum()} matches where winner = loser")
        issues.append(f"Winner equals loser: {same_player.sum()}")
    else:
        print("   ✅ No matches where winner = loser")
    
    # 3. Check ELO consistency
    print("\n3️⃣ ELO RATINGS:")
    # After match, winner ELO should be > before (usually)
    # After match, loser ELO should be < before (usually)
    
    if 'winner_elo_before' in matches.columns:
        winner_elo_down = matches['winner_elo_after'] < matches['winner_elo_before']
        loser_elo_up = matches['loser_elo_after'] > matches['loser_elo_before']
        
        print(f"   Winner ELO decreased: {winner_elo_down.sum():,} matches ({winner_elo_down.sum()/len(matches)*100:.1f}%)")
        print(f"   Loser ELO increased: {loser_elo_up.sum():,} matches ({loser_elo_up.sum()/len(matches)*100:.1f}%)")
        
        # This is actually WRONG - should be opposite!
        if winner_elo_down.sum() > 0:
            print(f"   ⚠️  WARNING: Winner's ELO should always increase!")
            issues.append(f"Winner ELO decreased: {winner_elo_down.sum()}")
        
        # Check for extreme ELO values
        extreme_elo = (matches['winner_elo_before'] < 1000) | (matches['winner_elo_before'] > 3000)
        extreme_elo |= (matches['loser_elo_before'] < 1000) | (matches['loser_elo_before'] > 3000)
        if extreme_elo.sum() > 0:
            print(f"   ⚠️  {extreme_elo.sum()} matches with extreme ELO (<1000 or >3000)")
            issues.append(f"Extreme ELO values: {extreme_elo.sum()}")
    else:
        print("   ⚠️  ELO columns not found")
    
    # 4. Check ranking consistency
    print("\n4️⃣ RANKINGS:")
    if 'winner_rank' in matches.columns:
        invalid_rank = (matches['winner_rank'] < 1) | (matches['winner_rank'] > 5000)
        invalid_rank |= (matches['loser_rank'] < 1) | (matches['loser_rank'] > 5000)
        invalid_rank = invalid_rank & matches['winner_rank'].notna()
        
        if invalid_rank.sum() > 0:
            print(f"   ❌ {invalid_rank.sum()} matches with invalid rankings (<1 or >5000)")
            issues.append(f"Invalid rankings: {invalid_rank.sum()}")
        else:
            print("   ✅ All rankings in valid range (1-5000)")
    
    # 5. Check surface values
    print("\n5️⃣ SURFACE VALUES:")
    valid_surfaces = ['Hard', 'Clay', 'Grass', 'Carpet']
    if 'surface' in matches.columns:
        surface_counts = matches['surface'].value_counts(dropna=False)
        print("   Surface distribution:")
        for surface, count in surface_counts.items():
            status = "✅" if surface in valid_surfaces else "❌"
            print(f"      {status} {surface}: {count:,} ({count/len(matches)*100:.1f}%)")
        
        invalid_surfaces = ~matches['surface'].isin(valid_surfaces) & matches['surface'].notna()
        if invalid_surfaces.sum() > 0:
            print(f"   ❌ Found {invalid_surfaces.sum()} matches with invalid surfaces")
            issues.append(f"Invalid surfaces: {invalid_surfaces.sum()}")
    
    # 6. Check date consistency
    print("\n6️⃣ DATE CONSISTENCY:")
    matches['date'] = pd.to_datetime(matches['date'])
    future_dates = matches['date'] > datetime.now()
    if future_dates.sum() > 0:
        print(f"   ⚠️  {future_dates.sum()} matches with future dates")
        print(f"      Latest: {matches[future_dates]['date'].max()}")
    else:
        print("   ✅ No future dates")
    
    old_dates = matches['date'] < datetime(1968, 1, 1)  # ATP started 1968
    if old_dates.sum() > 0:
        print(f"   ❌ {old_dates.sum()} matches before ATP era (1968)")
        issues.append(f"Matches before 1968: {old_dates.sum()}")
    else:
        print("   ✅ All dates after ATP era")
    
    # 7. Check for missing critical fields
    print("\n7️⃣ MISSING CRITICAL FIELDS:")
    critical = ['winner_name', 'loser_name', 'tournament', 'surface', 'round']
    for field in critical:
        missing = matches[field].isna().sum()
        if missing > 0:
            print(f"   ❌ {field}: {missing:,} missing ({missing/len(matches)*100:.1f}%)")
            issues.append(f"Missing {field}: {missing}")
        else:
            print(f"   ✅ {field}: Complete")
    
    # 8. Check for impossible service statistics
    print("\n8️⃣ SERVICE STATISTICS VALIDATION:")
    if 'w_ace' in matches.columns:
        # Aces should be reasonable (0-50 typically)
        extreme_aces = (matches['w_ace'] > 50) & matches['w_ace'].notna()
        if extreme_aces.sum() > 0:
            print(f"   ⚠️  {extreme_aces.sum()} matches with >50 aces (extreme but possible)")
            max_aces = matches.loc[matches['w_ace'].notna(), 'w_ace'].max()
            print(f"      Max aces in dataset: {max_aces}")
        
        # Check for negative values
        negative_stats = (matches['w_ace'] < 0) | (matches['w_df'] < 0)
        negative_stats = negative_stats & matches['w_ace'].notna()
        if negative_stats.sum() > 0:
            print(f"   ❌ {negative_stats.sum()} matches with negative service stats")
            issues.append(f"Negative service stats: {negative_stats.sum()}")
        else:
            print("   ✅ No negative service statistics")
    
    # 9. Check odds consistency
    print("\n9️⃣ BETTING ODDS:")
    if 'winner_odds' in matches.columns:
        has_odds = matches['winner_odds'].notna()
        print(f"   Matches with odds: {has_odds.sum():,} ({has_odds.sum()/len(matches)*100:.1f}%)")
        
        # Odds should be >= 1.0
        if has_odds.sum() > 0:
            invalid_odds = (matches['winner_odds'] < 1.0) | (matches['loser_odds'] < 1.0)
            invalid_odds = invalid_odds & has_odds
            if invalid_odds.sum() > 0:
                print(f"   ❌ {invalid_odds.sum()} matches with odds < 1.0")
                issues.append(f"Invalid odds: {invalid_odds.sum()}")
            else:
                print("   ✅ All odds >= 1.0")
            
            # Check if winner always has lower odds (should be favorite)
            winner_underdog = (matches['winner_odds'] > matches['loser_odds']) & has_odds
            print(f"   Upsets (winner was underdog): {winner_underdog.sum():,} ({winner_underdog.sum()/has_odds.sum()*100:.1f}%)")
    
    # 10. Check player name consistency
    print("\n🔟 PLAYER NAME FORMAT:")
    # Check for unusual characters or formats
    if 'winner_name' in matches.columns:
        # Check for very long names (>50 chars)
        long_names = matches['winner_name'].str.len() > 50
        long_names |= matches['loser_name'].str.len() > 50
        if long_names.sum() > 0:
            print(f"   ⚠️  {long_names.sum()} matches with very long player names (>50 chars)")
        
        # Check for names with numbers
        has_numbers = matches['winner_name'].str.contains(r'\d', na=False)
        has_numbers |= matches['loser_name'].str.contains(r'\d', na=False)
        if has_numbers.sum() > 0:
            print(f"   ⚠️  {has_numbers.sum()} matches with numbers in player names")
            sample = matches[has_numbers].head(3)[['winner_name', 'loser_name']]
            print(sample.to_string(index=False))
        else:
            print("   ✅ No numbers in player names")
    
    return issues


def check_players_data():
    """Check players.csv for inconsistencies"""
    print("\n\n" + "="*70)
    print("PLAYERS.CSV - DATA QUALITY CHECK")
    print("="*70)
    
    players = pd.read_csv('data/processed/players.csv')
    issues = []
    
    print(f"\n📊 Dataset: {len(players):,} players")
    
    # 1. Check for duplicate players
    print("\n1️⃣ DUPLICATE PLAYERS:")
    duplicates = players['player_name'].duplicated()
    if duplicates.sum() > 0:
        print(f"   ❌ Found {duplicates.sum()} duplicate player names")
        issues.append(f"Duplicate players: {duplicates.sum()}")
        print(players[duplicates]['player_name'].head(10).tolist())
    else:
        print("   ✅ No duplicate player names")
    
    # 2. Check win percentage logic
    print("\n2️⃣ WIN PERCENTAGE:")
    if 'win_percentage' in players.columns:
        invalid_pct = (players['win_percentage'] < 0) | (players['win_percentage'] > 100)
        if invalid_pct.sum() > 0:
            print(f"   ❌ {invalid_pct.sum()} players with invalid win % (<0 or >100)")
            issues.append(f"Invalid win percentage: {invalid_pct.sum()}")
        else:
            print("   ✅ All win percentages in valid range")
        
        # Check if matches total_wins/total_matches
        players['calc_win_pct'] = (players['total_wins'] / players['total_matches'] * 100).round(2)
        mismatch = (abs(players['win_percentage'] - players['calc_win_pct']) > 0.1) & players['total_matches'] > 0
        if mismatch.sum() > 0:
            print(f"   ⚠️  {mismatch.sum()} players with win % mismatch (calculated vs stored)")
            issues.append(f"Win percentage mismatch: {mismatch.sum()}")
    
    # 3. Check ELO ranges
    print("\n3️⃣ ELO RATINGS:")
    if 'final_elo' in players.columns:
        elo_stats = players['final_elo'].describe()
        print(f"   Range: {elo_stats['min']:.0f} to {elo_stats['max']:.0f}")
        print(f"   Mean: {elo_stats['mean']:.0f}, Median: {elo_stats['50%']:.0f}")
        
        extreme_low = players['final_elo'] < 1000
        extreme_high = players['final_elo'] > 3000
        if extreme_low.sum() > 0:
            print(f"   ⚠️  {extreme_low.sum()} players with ELO < 1000")
        if extreme_high.sum() > 0:
            print(f"   ⚠️  {extreme_high.sum()} players with ELO > 3000")
        if extreme_low.sum() == 0 and extreme_high.sum() == 0:
            print("   ✅ All ELOs in reasonable range (1000-3000)")
    
    # 4. Check physical stats
    print("\n4️⃣ PHYSICAL STATISTICS:")
    if 'height' in players.columns:
        valid_height = (players['height'] >= 150) & (players['height'] <= 220)
        invalid = ~valid_height & players['height'].notna()
        if invalid.sum() > 0:
            print(f"   ❌ {invalid.sum()} players with invalid height (<150cm or >220cm)")
            issues.append(f"Invalid heights: {invalid.sum()}")
        else:
            print(f"   ✅ All heights in valid range (150-220cm)")
    
    # 5. Check for players with 0 matches
    print("\n5️⃣ MATCH COUNTS:")
    no_matches = players['total_matches'] == 0
    if no_matches.sum() > 0:
        print(f"   ⚠️  {no_matches.sum()} players with 0 matches")
        issues.append(f"Players with 0 matches: {no_matches.sum()}")
    else:
        print("   ✅ All players have at least 1 match")
    
    return issues


def check_stadiums_data():
    """Check stadiums.csv for inconsistencies"""
    print("\n\n" + "="*70)
    print("STADIUMS.CSV - DATA QUALITY CHECK")
    print("="*70)
    
    stadiums = pd.read_csv('data/processed/stadiums.csv')
    issues = []
    
    print(f"\n📊 Dataset: {len(stadiums):,} tournament/surface combinations")
    
    # 1. Check for duplicate tournaments
    print("\n1️⃣ DUPLICATE TOURNAMENTS:")
    duplicates = stadiums.duplicated(subset=['tournament', 'surface'])
    if duplicates.sum() > 0:
        print(f"   ❌ Found {duplicates.sum()} duplicate tournament/surface combos")
        issues.append(f"Duplicate tournaments: {duplicates.sum()}")
    else:
        print("   ✅ No duplicate tournament/surface combinations")
    
    # 2. Check date logic
    print("\n2️⃣ DATE LOGIC:")
    stadiums['first_date'] = pd.to_datetime(stadiums['first_date'])
    stadiums['last_date'] = pd.to_datetime(stadiums['last_date'])
    
    invalid_dates = stadiums['first_date'] > stadiums['last_date']
    if invalid_dates.sum() > 0:
        print(f"   ❌ {invalid_dates.sum()} tournaments where first_date > last_date")
        issues.append(f"Invalid date logic: {invalid_dates.sum()}")
    else:
        print("   ✅ All tournaments have valid date ranges")
    
    # 3. Check location coverage
    print("\n3️⃣ LOCATION COVERAGE:")
    has_country = stadiums['country'].notna()
    print(f"   Tournaments with country: {has_country.sum():,} ({has_country.sum()/len(stadiums)*100:.1f}%)")
    
    various = stadiums['country'] == 'Various'
    print(f"   Multi-location events (Various): {various.sum():,}")
    
    real_locations = has_country & ~various
    print(f"   Tournaments with specific location: {real_locations.sum():,} ({real_locations.sum()/len(stadiums)*100:.1f}%)")
    
    return issues


def check_cross_file_consistency():
    """Check consistency across files"""
    print("\n\n" + "="*70)
    print("CROSS-FILE CONSISTENCY CHECK")
    print("="*70)
    
    matches = pd.read_csv('data/processed/matches.csv', low_memory=False)
    players = pd.read_csv('data/processed/players.csv')
    stadiums = pd.read_csv('data/processed/stadiums.csv')
    
    issues = []
    
    # 1. Check if all players in matches exist in players.csv
    print("\n1️⃣ PLAYER CONSISTENCY:")
    winners_in_matches = set(matches['winner_name'].dropna().unique())
    losers_in_matches = set(matches['loser_name'].dropna().unique())
    all_players_in_matches = winners_in_matches | losers_in_matches
    
    players_in_players_file = set(players['player_name'].unique())
    
    missing_from_players = all_players_in_matches - players_in_players_file
    if len(missing_from_players) > 0:
        print(f"   ❌ {len(missing_from_players)} players in matches.csv missing from players.csv")
        issues.append(f"Players missing from players.csv: {len(missing_from_players)}")
        print(f"      Sample: {list(missing_from_players)[:5]}")
    else:
        print("   ✅ All players in matches.csv exist in players.csv")
    
    extra_players = players_in_players_file - all_players_in_matches
    if len(extra_players) > 0:
        print(f"   ⚠️  {len(extra_players)} players in players.csv with no matches")
    
    # 2. Check tournament consistency
    print("\n2️⃣ TOURNAMENT CONSISTENCY:")
    tournaments_in_matches = matches.groupby(['tournament', 'surface']).size().reset_index()[['tournament', 'surface']]
    stadiums_combo = stadiums[['tournament', 'surface']]
    
    # Merge to find mismatches
    merged = tournaments_in_matches.merge(
        stadiums_combo,
        on=['tournament', 'surface'],
        how='left',
        indicator=True
    )
    
    missing_from_stadiums = merged[merged['_merge'] == 'left_only']
    if len(missing_from_stadiums) > 0:
        print(f"   ❌ {len(missing_from_stadiums)} tournament/surface combos in matches.csv missing from stadiums.csv")
        issues.append(f"Tournaments missing from stadiums.csv: {len(missing_from_stadiums)}")
    else:
        print("   ✅ All tournaments in matches.csv exist in stadiums.csv")
    
    return issues


def main():
    print("="*70)
    print("🔍 COMPREHENSIVE DATA QUALITY CHECK")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    all_issues = []
    
    # Check each file
    all_issues.extend(check_matches_data())
    all_issues.extend(check_players_data())
    all_issues.extend(check_stadiums_data())
    all_issues.extend(check_cross_file_consistency())
    
    # Final summary
    print("\n\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    
    if len(all_issues) == 0:
        print("\n✅ NO CRITICAL ISSUES FOUND!")
        print("   Data quality is excellent!")
    else:
        print(f"\n⚠️  FOUND {len(all_issues)} ISSUES:")
        for i, issue in enumerate(all_issues, 1):
            print(f"   {i}. {issue}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
