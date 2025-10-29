# New Features to Add - Enhancement Plan

## 🎯 High Priority Features (Should Definitely Add)

### 1. Betting Market Features (NEW DATA SOURCE!)
**Source:** ATP Tennis 2000-2025 Daily Update has betting odds!

```python
# From Odd_1 and Odd_2 columns
- implied_probability_p1        # 1 / Odd_1
- implied_probability_p2        # 1 / Odd_2
- odds_diff                     # Odd_1 - Odd_2
- odds_ratio                    # Odd_1 / Odd_2
- market_favorite               # Binary: is player1 favorite?
- odds_elo_agreement            # Do odds agree with our ELO prediction?
- odds_elo_divergence           # How much do they differ?
- market_confidence             # Lower odds = higher market confidence
- value_bet_p1                  # Our probability vs market probability
- value_bet_p2                  # Our probability vs market probability
```

**Why Important:**
- Betting markets aggregate expert opinions
- Can calibrate our predictions against real-world probabilities
- Could boost accuracy by 2-5%
- Helps identify when our model disagrees with consensus

---

### 2. Enhanced Ranking Features
**Source:** Jeff Sackmann has ranking_date, rank, points

```python
# Current features only use rank at match time
- player1_ranking_trend_30d     # Is ranking improving/declining?
- player2_ranking_trend_30d     
- player1_ranking_trend_90d     # Longer term trend
- player2_ranking_trend_90d
- ranking_points_diff           # More granular than rank
- ranking_volatility_p1         # How stable is their ranking?
- ranking_volatility_p2
- career_high_rank_p1           # Peak ranking
- career_high_rank_p2
- weeks_at_current_rank_p1      # Consistency
- weeks_at_current_rank_p2
- rank_momentum_diff            # Combined ranking trajectory
```

**Why Important:**
- A player rising from #50 to #30 is different from falling #30 to #50
- Captures player trajectory/momentum
- Old model only used static rank at match time

---

### 3. Advanced Surface Specialization
**Current:** Binary specialist flag (not used by model!)

```python
# New approach with granularity
- player1_surface_win_pct       # Win % on this surface
- player2_surface_win_pct
- player1_surface_matches       # Experience on surface
- player2_surface_matches
- surface_experience_diff       # Who's more experienced?
- player1_recent_surface_form   # Last 10 matches on this surface
- player2_recent_surface_form
- surface_transition            # Coming from different surface?
- player1_clay_hard_ratio       # Surface versatility
- player2_clay_hard_ratio
- indoor_outdoor_factor         # Court environment
```

**Why Important:**
- Some players dominate on clay but struggle on hard
- Better than binary specialist flag
- Accounts for surface transitions (clay season → hard court)

---

### 4. Match Context & Scheduling
**Source:** Available from tourney data

```python
# Tournament progression
- round_number                  # R32=1, R16=2, QF=3, SF=4, F=5
- matches_to_title              # How many wins needed?
- prize_money_difference        # Stakes of this match
- knockout_pressure             # Grand Slam vs ATP 250

# Scheduling pressure
- matches_in_last_7_days        # Tournament congestion
- matches_in_last_14_days
- back_to_back_days             # Played yesterday?
- triple_threat_week            # 3 matches in week?
- travel_burden                 # Different continent last week?

# Tournament phase
- early_round                   # Round 1-2 (upsets more likely)
- late_round                    # QF, SF, F (favorites stronger)
- defending_champion_p1         # Extra pressure
- defending_champion_p2
```

**Why Important:**
- Favorites are stronger in later rounds
- Scheduling density affects performance
- Context matters: Grand Slam final ≠ ATP 250 first round

---

### 5. Enhanced Head-to-Head
**Current:** Basic H2H wins/matches (low importance in old model)

```python
# Make H2H more useful
- h2h_on_this_surface           # H2H record on current surface
- h2h_in_last_year              # Recent H2H (more relevant)
- h2h_at_this_tournament        # Venue-specific H2H
- h2h_streak                    # Current win streak in H2H
- h2h_decisive_sets             # How close are their matches?
- h2h_avg_set_diff              # Dominance level
- h2h_recency_weighted          # Weight recent matches more
- psychological_edge            # Consistent winner in H2H
```

**Why Important:**
- Some matchups are psychological (Nadal vs Federer on clay)
- Recent H2H more relevant than 5 years ago
- Surface-specific H2H is key

---

### 6. Service & Return Statistics
**Source:** Jeff Sackmann has w_ace, w_df, w_svpt, w_1stIn, etc.

```python
# Service game strength
- player1_avg_aces_per_match    # Rolling average
- player2_avg_aces_per_match
- player1_avg_df_per_match      # Double faults
- player2_avg_df_per_match
- player1_1st_serve_pct         # Service consistency
- player2_1st_serve_pct
- player1_1st_serve_win_pct     # Service effectiveness
- player2_1st_serve_win_pct
- player1_break_points_saved    # Pressure handling
- player2_break_points_saved

# Return game strength  
- player1_return_points_won     # Return ability
- player2_return_points_won
- player1_break_points_converted
- player2_break_points_converted

# Matchup advantages
- service_game_diff             # Who has better serve?
- return_game_diff              # Who returns better?
- serve_return_balance          # Overall advantage
```

**Why Important:**
- Tennis is won on serve/return
- Big servers have advantage on fast courts
- Good returners neutralize big servers
- Surface impacts these stats

---

### 7. Match Duration & Stamina
**Source:** minutes column in Jeff Sackmann data

```python
- player1_avg_match_duration    # How long their matches last
- player2_avg_match_duration
- player1_recent_marathon       # Recently played 3+ hour match?
- player2_recent_marathon
- player1_total_sets_played     # Volume in last 2 weeks
- player2_total_sets_played
- stamina_advantage             # Who's fresher?
- best_of_5_experience          # Grand Slam stamina
- tiebreak_record_p1            # Clutch performance
- tiebreak_record_p2
```

**Why Important:**
- Marathon matches affect recovery
- Some players excel in long matches
- Best-of-5 requires different stamina

---

### 8. Career Stage & Experience
**Source:** Can derive from age and match history

```python
- player1_career_matches        # Total career matches
- player2_career_matches
- experience_gap                # Experience difference
- player1_prime_age             # Age 24-29 typically peak
- player2_prime_age
- veteran_factor_p1             # Age 32+ with experience
- veteran_factor_p2
- rising_star_p1                # Young + improving rank
- rising_star_p2
- comeback_from_injury_p1       # Recently returned from break
- comeback_from_injury_p2
```

**Why Important:**
- Experience matters in big moments
- Age affects but isn't linear (old model used age_diff, had 0 importance)
- Prime years (25-29) are peak performance

---

## 🎲 Medium Priority Features (Worth Testing)

### 9. Time of Year / Season Context
```python
- month_of_year                 # 1-12
- season_phase                  # Start/Mid/End of season
- clay_court_season             # April-June
- hard_court_season             # Jan-Mar, Aug-Oct
- grass_court_season            # June-July
- surface_season_match          # Playing surface in its season
```

### 10. Tournament Prestige & Stakes
```python
- tournament_total_prize_money  
- ranking_points_for_winner
- prestige_score                # Grand Slam=4, Masters=3, etc.
- home_tournament_p1            # Playing in home country
- home_tournament_p2
```

### 11. Weather & Court Conditions (if available)
```python
- court_speed_index             # Fast/Medium/Slow
- altitude                      # High altitude = faster
- indoor_factor                 # Different ball behavior
- temperature                   # Heat affects performance
```

---

## ❌ Features to Remove/Skip

Based on old model analysis (zero importance):

```python
# Don't bother with these:
- player_height                 # Had ZERO importance
- height_difference
- player_taller
- handedness features           # All had ZERO importance
- lefty indicators
- age_diff (static)             # Use age trajectory instead
```

---

## 🎯 Feature Priority Ranking

### Must Add (Top 10):
1. ✅ **Betting odds features** (10 features) - NEW DATA!
2. ✅ **Ranking trends** (8 features) - Momentum matters
3. ✅ **Surface win %** (6 features) - Better than binary specialist
4. ✅ **Service statistics** (10 features) - Core of tennis
5. ✅ **Enhanced H2H** (8 features) - Make it useful
6. ✅ **Match context** (6 features) - Round matters
7. ✅ **Return statistics** (6 features) - Balance with serve
8. ✅ **Ranking points** (4 features) - More granular
9. ✅ **Recent surface form** (4 features) - Surface-specific
10. ✅ **Tournament progression** (4 features) - Context

**New Feature Count: ~66 features**

### Should Add (Next 10):
11. Match duration/stamina (6 features)
12. Career stage (6 features)
13. Scheduling pressure (5 features)
14. Surface transition (4 features)
15. Tournament prestige (4 features)
16. Season context (4 features)
17. Tiebreak records (2 features)
18. Defending champion (2 features)

**Additional: ~33 features**

---

## 📊 Expected Total Features

- **Old features to keep:** ~96 features (remove 14 zero-importance ones)
- **New priority features:** ~66 features
- **Medium priority features:** ~33 features

**Total: ~195 features** (can prune after seeing importance)

---

## 🚀 Implementation Strategy

### Phase 1: Core Enhancement (Week 1)
1. Add betting odds features (immediate boost expected)
2. Add ranking trends
3. Add service/return stats
4. Test model - expect 67-70% accuracy

### Phase 2: Context & Specialization (Week 2)
5. Add surface specialization
6. Add match context
7. Add enhanced H2H
8. Test model - expect 70-72% accuracy

### Phase 3: Fine-Tuning (Week 3)
9. Add medium priority features
10. Feature importance analysis
11. Remove low-importance features
12. Final tuning - target 72-75% accuracy

---

## 💡 Key Insights

1. **Betting odds** = biggest opportunity (we have this data!)
2. **Dynamic features** > static features (trends vs snapshots)
3. **Context matters** (round, surface, tournament level)
4. **Remove zero-importance features** (height, handedness)
5. **Service/return** = core tennis metrics (surprisingly not in old model!)

---

## 🎯 Expected Accuracy Improvement

| Phase | Features | Expected Accuracy |
|-------|----------|-------------------|
| Old Model | 110 | 64.9% |
| Phase 1 (Betting + Stats) | 150 | 67-70% |
| Phase 2 (Context + Surface) | 180 | 70-72% |
| Phase 3 (Full + Tuned) | 150-180 | 72-75% |

**Conservative estimate:** 68-70% accuracy
**Optimistic estimate:** 72-75% accuracy

With clean data + new features, we should break the 65% ceiling! 🎾
