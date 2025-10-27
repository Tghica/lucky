import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class EloCalculator:
    """Elo rating system for tennis players with surface-specific and tournament-specific ratings."""
    
    def __init__(self, initial_rating: int = 1500, k_factor: int = 32):
        """
        Initialize Elo calculator.
        
        Args:
            initial_rating: Starting Elo rating for all players
            k_factor: Maximum rating change per match (higher = more volatile)
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.player_ratings: Dict[str, float] = {}  # General Elo
        self.player_ratings_hard: Dict[str, float] = {}  # Hard court Elo
        self.player_ratings_clay: Dict[str, float] = {}  # Clay court Elo
        self.player_ratings_grass: Dict[str, float] = {}  # Grass court Elo
        self.player_ratings_carpet: Dict[str, float] = {}  # Carpet court Elo
        
        # Tournament-specific Elo
        self.player_ratings_grand_slam: Dict[str, float] = {}  # Grand Slam Elo
        self.player_ratings_masters: Dict[str, float] = {}  # Masters 1000 Elo
        self.player_ratings_atp500: Dict[str, float] = {}  # ATP 500 Elo
        self.player_ratings_atp250: Dict[str, float] = {}  # ATP 250 Elo
    
    @staticmethod
    def classify_tournament(tournament_name: str) -> str:
        """Classify tournament into categories."""
        tournament_lower = tournament_name.lower()
        
        # Grand Slams
        if any(gs in tournament_lower for gs in ['australian open', 'french open', 'roland garros', 'wimbledon', 'us open']):
            return 'grand_slam'
        
        # Masters 1000 (various names over the years)
        masters_keywords = [
            'masters', 'indian wells', 'bnp paribas open', 'miami open', 'sony ericsson',
            'monte carlo', 'monte-carlo', 'madrid', 'rome', 'internazionali',
            'canada masters', 'rogers', 'cincinnati', 'western & southern',
            'shanghai', 'paris masters', 'bercy'
        ]
        if any(kw in tournament_lower for kw in masters_keywords):
            return 'masters'
        
        # ATP 500 (common 500-level tournaments)
        atp500_keywords = [
            'barcelona', 'london', 'dubai', 'rotterdam', 'abm amro',
            'rio', 'acapulco', 'mexicano', 'washington', 'hamburg',
            'beijing', 'tokyo', 'basel', 'vienna', 'queens'
        ]
        if any(kw in tournament_lower for kw in atp500_keywords):
            return 'atp500'
        
        # Everything else is ATP 250 or lower
        return 'atp250'
    
    def get_rating(self, player: str, surface: str = None, tournament_level: str = None) -> float:
        """Get current rating for a player, initialize if new."""
        if tournament_level:
            # Tournament-specific Elo
            if tournament_level == 'grand_slam':
                ratings_dict = self.player_ratings_grand_slam
            elif tournament_level == 'masters':
                ratings_dict = self.player_ratings_masters
            elif tournament_level == 'atp500':
                ratings_dict = self.player_ratings_atp500
            else:  # atp250
                ratings_dict = self.player_ratings_atp250
            
            if player not in ratings_dict:
                ratings_dict[player] = self.initial_rating
            return ratings_dict[player]
        elif surface is None:
            # General Elo
            if player not in self.player_ratings:
                self.player_ratings[player] = self.initial_rating
            return self.player_ratings[player]
        else:
            # Surface-specific Elo
            surface_lower = surface.lower()
            if surface_lower == 'hard':
                ratings_dict = self.player_ratings_hard
            elif surface_lower == 'clay':
                ratings_dict = self.player_ratings_clay
            elif surface_lower == 'grass':
                ratings_dict = self.player_ratings_grass
            elif surface_lower == 'carpet':
                ratings_dict = self.player_ratings_carpet
            else:
                ratings_dict = self.player_ratings_hard  # Default to hard
            
            if player not in ratings_dict:
                ratings_dict[player] = self.initial_rating
            return ratings_dict[player]
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score (win probability) for player A."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, winner: str, loser: str, surface: str, tournament_level: str) -> Tuple:
        """
        Update general, surface-specific, and tournament-specific Elo ratings after a match.
        
        Returns:
            Tuple of (winner_old_gen, winner_new_gen, loser_old_gen, loser_new_gen,
                     winner_old_surf, winner_new_surf, loser_old_surf, loser_new_surf,
                     winner_old_tourn, winner_new_tourn, loser_old_tourn, loser_new_tourn)
        """
        # Update general Elo
        winner_old_gen = self.get_rating(winner)
        loser_old_gen = self.get_rating(loser)
        
        winner_expected_gen = self.expected_score(winner_old_gen, loser_old_gen)
        loser_expected_gen = self.expected_score(loser_old_gen, winner_old_gen)
        
        winner_new_gen = winner_old_gen + self.k_factor * (1 - winner_expected_gen)
        loser_new_gen = loser_old_gen + self.k_factor * (0 - loser_expected_gen)
        
        self.player_ratings[winner] = winner_new_gen
        self.player_ratings[loser] = loser_new_gen
        
        # Update surface-specific Elo
        winner_old_surf = self.get_rating(winner, surface)
        loser_old_surf = self.get_rating(loser, surface)
        
        winner_expected_surf = self.expected_score(winner_old_surf, loser_old_surf)
        loser_expected_surf = self.expected_score(loser_old_surf, winner_old_surf)
        
        winner_new_surf = winner_old_surf + self.k_factor * (1 - winner_expected_surf)
        loser_new_surf = loser_old_surf + self.k_factor * (0 - loser_expected_surf)
        
        # Store surface-specific ratings
        surface_lower = surface.lower()
        if surface_lower == 'hard':
            self.player_ratings_hard[winner] = winner_new_surf
            self.player_ratings_hard[loser] = loser_new_surf
        elif surface_lower == 'clay':
            self.player_ratings_clay[winner] = winner_new_surf
            self.player_ratings_clay[loser] = loser_new_surf
        elif surface_lower == 'grass':
            self.player_ratings_grass[winner] = winner_new_surf
            self.player_ratings_grass[loser] = loser_new_surf
        elif surface_lower == 'carpet':
            self.player_ratings_carpet[winner] = winner_new_surf
            self.player_ratings_carpet[loser] = loser_new_surf
        
        # Update tournament-specific Elo
        winner_old_tourn = self.get_rating(winner, tournament_level=tournament_level)
        loser_old_tourn = self.get_rating(loser, tournament_level=tournament_level)
        
        winner_expected_tourn = self.expected_score(winner_old_tourn, loser_old_tourn)
        loser_expected_tourn = self.expected_score(loser_old_tourn, winner_old_tourn)
        
        winner_new_tourn = winner_old_tourn + self.k_factor * (1 - winner_expected_tourn)
        loser_new_tourn = loser_old_tourn + self.k_factor * (0 - loser_expected_tourn)
        
        # Store tournament-specific ratings
        if tournament_level == 'grand_slam':
            self.player_ratings_grand_slam[winner] = winner_new_tourn
            self.player_ratings_grand_slam[loser] = loser_new_tourn
        elif tournament_level == 'masters':
            self.player_ratings_masters[winner] = winner_new_tourn
            self.player_ratings_masters[loser] = loser_new_tourn
        elif tournament_level == 'atp500':
            self.player_ratings_atp500[winner] = winner_new_tourn
            self.player_ratings_atp500[loser] = loser_new_tourn
        else:  # atp250
            self.player_ratings_atp250[winner] = winner_new_tourn
            self.player_ratings_atp250[loser] = loser_new_tourn
        
        return (winner_old_gen, winner_new_gen, loser_old_gen, loser_new_gen,
                winner_old_surf, winner_new_surf, loser_old_surf, loser_new_surf,
                winner_old_tourn, winner_new_tourn, loser_old_tourn, loser_new_tourn)
    
    def calculate_match_elos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate general, surface-specific, and tournament-specific Elo ratings for all matches.
        
        Adds columns for Elo ratings before/after matches, differences, and win probabilities.
        """
        # Sort by date to process chronologically
        df = df.sort_values('date').reset_index(drop=True)
        
        # Initialize new columns for general Elo
        df['player1_elo_before'] = 0.0
        df['player1_elo_after'] = 0.0
        df['player2_elo_before'] = 0.0
        df['player2_elo_after'] = 0.0
        df['elo_diff'] = 0.0
        df['win_probability'] = 0.0
        
        # Initialize new columns for surface-specific Elo
        df['player1_surface_elo_before'] = 0.0
        df['player1_surface_elo_after'] = 0.0
        df['player2_surface_elo_before'] = 0.0
        df['player2_surface_elo_after'] = 0.0
        df['surface_elo_diff'] = 0.0
        df['surface_win_probability'] = 0.0
        
        # Initialize new columns for tournament-specific Elo
        df['player1_tournament_elo_before'] = 0.0
        df['player1_tournament_elo_after'] = 0.0
        df['player2_tournament_elo_before'] = 0.0
        df['player2_tournament_elo_after'] = 0.0
        df['tournament_elo_diff'] = 0.0
        df['tournament_win_probability'] = 0.0
        df['tournament_level'] = ''
        
        for idx, row in df.iterrows():
            player1 = row['player1']
            player2 = row['player2']
            winner = row['winner']
            surface = row['surface']
            tournament_name = row.get('description', '')
            
            # Classify tournament level
            tournament_level = self.classify_tournament(tournament_name)
            df.at[idx, 'tournament_level'] = tournament_level
            
            # Get current ratings
            p1_elo_gen = self.get_rating(player1)
            p2_elo_gen = self.get_rating(player2)
            p1_elo_surf = self.get_rating(player1, surface)
            p2_elo_surf = self.get_rating(player2, surface)
            p1_elo_tourn = self.get_rating(player1, tournament_level=tournament_level)
            p2_elo_tourn = self.get_rating(player2, tournament_level=tournament_level)
            
            # Calculate probabilities
            win_prob_gen = self.expected_score(p1_elo_gen, p2_elo_gen)
            win_prob_surf = self.expected_score(p1_elo_surf, p2_elo_surf)
            win_prob_tourn = self.expected_score(p1_elo_tourn, p2_elo_tourn)
            
            # Store before ratings
            df.at[idx, 'player1_elo_before'] = p1_elo_gen
            df.at[idx, 'player2_elo_before'] = p2_elo_gen
            df.at[idx, 'elo_diff'] = p1_elo_gen - p2_elo_gen
            df.at[idx, 'win_probability'] = win_prob_gen
            
            df.at[idx, 'player1_surface_elo_before'] = p1_elo_surf
            df.at[idx, 'player2_surface_elo_before'] = p2_elo_surf
            df.at[idx, 'surface_elo_diff'] = p1_elo_surf - p2_elo_surf
            df.at[idx, 'surface_win_probability'] = win_prob_surf
            
            df.at[idx, 'player1_tournament_elo_before'] = p1_elo_tourn
            df.at[idx, 'player2_tournament_elo_before'] = p2_elo_tourn
            df.at[idx, 'tournament_elo_diff'] = p1_elo_tourn - p2_elo_tourn
            df.at[idx, 'tournament_win_probability'] = win_prob_tourn
            
            # Determine loser
            loser = player2 if winner == player1 else player1
            
            # Update ratings and get new values
            (w_old_gen, w_new_gen, l_old_gen, l_new_gen,
             w_old_surf, w_new_surf, l_old_surf, l_new_surf,
             w_old_tourn, w_new_tourn, l_old_tourn, l_new_tourn) = self.update_ratings(winner, loser, surface, tournament_level)
            
            # Store after ratings
            if winner == player1:
                df.at[idx, 'player1_elo_after'] = w_new_gen
                df.at[idx, 'player2_elo_after'] = l_new_gen
                df.at[idx, 'player1_surface_elo_after'] = w_new_surf
                df.at[idx, 'player2_surface_elo_after'] = l_new_surf
                df.at[idx, 'player1_tournament_elo_after'] = w_new_tourn
                df.at[idx, 'player2_tournament_elo_after'] = l_new_tourn
            else:
                df.at[idx, 'player1_elo_after'] = l_new_gen
                df.at[idx, 'player2_elo_after'] = w_new_gen
                df.at[idx, 'player1_surface_elo_after'] = l_new_surf
                df.at[idx, 'player2_surface_elo_after'] = w_new_surf
                df.at[idx, 'player1_tournament_elo_after'] = l_new_tourn
                df.at[idx, 'player2_tournament_elo_after'] = w_new_tourn
        
        return df
    
    def calculate_form(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """
        Calculate player form based on recent match performance.
        Form is tracked as individual match outcomes for the last N matches.
        
        Args:
            df: DataFrame with match data (must be chronologically sorted)
            window: Number of recent matches to consider for form calculation (default: 10)
            
        Returns:
            DataFrame with added form columns:
            - player1_match_1 through player1_match_10: 1=Won, 0=Lost, NaN=No match
            - player2_match_1 through player2_match_10: 1=Won, 0=Lost, NaN=No match
            
        Note: match_1 is the most recent match, match_10 is the oldest
        """
        # Initialize form columns as lists (faster than df.at[])
        # Create dictionary to store match outcome lists
        player1_matches = {f'player1_match_{i}': [] for i in range(1, window + 1)}
        player2_matches = {f'player2_match_{i}': [] for i in range(1, window + 1)}
        
        # Track match history for each player using deque for efficiency
        from collections import deque
        player_match_history = {}  # {player: deque of is_win booleans (1=won, 0=lost)}
        
        for idx, row in df.iterrows():
            player1 = row['player1']
            player2 = row['player2']
            winner = row['winner']
            
            # Initialize history if not exists
            if player1 not in player_match_history:
                player_match_history[player1] = deque(maxlen=window)
            if player2 not in player_match_history:
                player_match_history[player2] = deque(maxlen=window)
            
            # Get form based on history before this match
            # Convert deque to list for indexing (index 0 = most recent, -1 = oldest)
            p1_history = list(player_match_history[player1])
            p2_history = list(player_match_history[player2])
            
            # Player 1 form: store individual match outcomes
            for i in range(1, window + 1):
                if i <= len(p1_history):
                    # Convert True/False to 1/0, access from most recent (index i-1)
                    player1_matches[f'player1_match_{i}'].append(int(p1_history[i - 1]))
                else:
                    # No match data available for this position
                    player1_matches[f'player1_match_{i}'].append(None)
            
            # Player 2 form: store individual match outcomes
            for i in range(1, window + 1):
                if i <= len(p2_history):
                    # Convert True/False to 1/0, access from most recent (index i-1)
                    player2_matches[f'player2_match_{i}'].append(int(p2_history[i - 1]))
                else:
                    # No match data available for this position
                    player2_matches[f'player2_match_{i}'].append(None)
            
            # Update match history after this match
            # Append to left to keep most recent first (index 0)
            player_match_history[player1].appendleft(winner == player1)
            player_match_history[player2].appendleft(winner == player2)
        
        # Add all match outcome columns to dataframe
        for i in range(1, window + 1):
            df[f'player1_match_{i}'] = player1_matches[f'player1_match_{i}']
            df[f'player2_match_{i}'] = player2_matches[f'player2_match_{i}']
        
        return df
    
    def calculate_head_to_head(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate head-to-head win rate for each player against their specific opponent.
        
        Args:
            df: DataFrame with match data (must be chronologically sorted)
            
        Returns:
            DataFrame with added head-to-head columns:
            - player1_h2h_wins: Number of wins player1 has against player2 before this match
            - player1_h2h_matches: Total matches between player1 and player2 before this match
            - player1_h2h_win_rate: Win rate of player1 against player2 (wins/matches)
            - player2_h2h_wins: Number of wins player2 has against player1 before this match
            - player2_h2h_matches: Total matches between player2 and player1 before this match
            - player2_h2h_win_rate: Win rate of player2 against player1 (wins/matches)
        """
        # Initialize head-to-head tracking
        # Structure: {(player_a, player_b): {'wins_a': int, 'wins_b': int, 'total': int}}
        h2h_records = {}
        
        # Initialize result columns as lists for efficiency
        p1_h2h_wins = []
        p1_h2h_matches = []
        p1_h2h_win_rate = []
        p2_h2h_wins = []
        p2_h2h_matches = []
        p2_h2h_win_rate = []
        
        for idx, row in df.iterrows():
            player1 = row['player1']
            player2 = row['player2']
            winner = row['winner']
            
            # Create a consistent key (alphabetically sorted to ensure same key for both orders)
            matchup_key = tuple(sorted([player1, player2]))
            
            # Initialize head-to-head record if this is first encounter
            if matchup_key not in h2h_records:
                h2h_records[matchup_key] = {
                    matchup_key[0]: 0,  # wins for first player alphabetically
                    matchup_key[1]: 0   # wins for second player alphabetically
                }
            
            # Get current h2h stats before this match
            total_matches = h2h_records[matchup_key][matchup_key[0]] + h2h_records[matchup_key][matchup_key[1]]
            p1_wins = h2h_records[matchup_key][player1]
            p2_wins = h2h_records[matchup_key][player2]
            
            # Calculate win rates (avoid division by zero)
            p1_win_rate = p1_wins / total_matches if total_matches > 0 else 0.0
            p2_win_rate = p2_wins / total_matches if total_matches > 0 else 0.0
            
            # Store stats for this match
            p1_h2h_wins.append(p1_wins)
            p1_h2h_matches.append(total_matches)
            p1_h2h_win_rate.append(p1_win_rate)
            p2_h2h_wins.append(p2_wins)
            p2_h2h_matches.append(total_matches)
            p2_h2h_win_rate.append(p2_win_rate)
            
            # Update head-to-head record after this match
            h2h_records[matchup_key][winner] += 1
        
        # Add columns to dataframe
        df['player1_h2h_wins'] = p1_h2h_wins
        df['player1_h2h_matches'] = p1_h2h_matches
        df['player1_h2h_win_rate'] = p1_h2h_win_rate
        df['player2_h2h_wins'] = p2_h2h_wins
        df['player2_h2h_matches'] = p2_h2h_matches
        df['player2_h2h_win_rate'] = p2_h2h_win_rate
        
        return df
    
    def calculate_fatigue(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate fatigue features: days since last match for each player.
        
        Players who played recently may be fatigued (disadvantage) or in good rhythm (advantage).
        Players with long breaks may be rusty or well-rested.
        
        Args:
            df: DataFrame with match data (must be chronologically sorted and have 'date' column)
            
        Returns:
            DataFrame with added fatigue columns:
            - player1_days_since_last: Days between this match and player1's previous match
            - player2_days_since_last: Days between this match and player2's previous match
            - rest_advantage: Difference in rest days (positive = player1 more rested)
            - player1_fatigued: 1 if player1 played within last 2 days, 0 otherwise
            - player2_fatigued: 1 if player2 played within last 2 days, 0 otherwise
            - both_rested: 1 if both players had 3+ days rest, 0 otherwise
        """
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Track last match date for each player
        player_last_match = {}  # {player: last_match_date}
        
        # Initialize result columns as lists
        p1_days_since_last = []
        p2_days_since_last = []
        rest_advantage = []
        p1_fatigued = []
        p2_fatigued = []
        both_rested = []
        
        for idx, row in df.iterrows():
            player1 = row['player1']
            player2 = row['player2']
            match_date = row['date']
            
            # Calculate days since last match for player1
            if player1 in player_last_match:
                days_since_p1 = (match_date - player_last_match[player1]).days
            else:
                days_since_p1 = None  # First match for this player
            
            # Calculate days since last match for player2
            if player2 in player_last_match:
                days_since_p2 = (match_date - player_last_match[player2]).days
            else:
                days_since_p2 = None  # First match for this player
            
            # Calculate rest advantage (None if either player has no history)
            if days_since_p1 is not None and days_since_p2 is not None:
                rest_adv = days_since_p1 - days_since_p2
            else:
                rest_adv = 0  # Neutral if no history
            
            # Fatigue flags (played within 2 days = fatigued)
            p1_is_fatigued = 1 if (days_since_p1 is not None and days_since_p1 < 2) else 0
            p2_is_fatigued = 1 if (days_since_p2 is not None and days_since_p2 < 2) else 0
            
            # Both well-rested flag (3+ days for both)
            both_well_rested = 1 if (
                days_since_p1 is not None and days_since_p1 >= 3 and
                days_since_p2 is not None and days_since_p2 >= 3
            ) else 0
            
            # Store results
            p1_days_since_last.append(days_since_p1 if days_since_p1 is not None else 7)  # Default to 7 days
            p2_days_since_last.append(days_since_p2 if days_since_p2 is not None else 7)
            rest_advantage.append(rest_adv)
            p1_fatigued.append(p1_is_fatigued)
            p2_fatigued.append(p2_is_fatigued)
            both_rested.append(both_well_rested)
            
            # Update last match date for both players
            player_last_match[player1] = match_date
            player_last_match[player2] = match_date
        
        # Add columns to dataframe
        df['player1_days_since_last'] = p1_days_since_last
        df['player2_days_since_last'] = p2_days_since_last
        df['rest_advantage'] = rest_advantage
        df['player1_fatigued'] = p1_fatigued
        df['player2_fatigued'] = p2_fatigued
        df['both_rested'] = both_rested
        
        logger.info("Calculated fatigue features: days since last match, rest advantage, fatigue flags")
        
        return df
    
    def calculate_tournament_progression(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate tournament progression features: matches played in current tournament.
        
        Players deep in a tournament may be fatigued OR in excellent form.
        This feature captures both endurance and momentum within a tournament.
        
        Args:
            df: DataFrame with match data (must be chronologically sorted with 'tournament' column)
            
        Returns:
            DataFrame with added tournament progression columns:
            - player1_matches_in_tournament: Number of matches player1 has played in this tournament
            - player2_matches_in_tournament: Number of matches player2 has played in this tournament
            - tournament_experience_diff: Difference (player1 - player2) in tournament matches
            - player1_deep_run: 1 if player1 has played 3+ matches in tournament, 0 otherwise
            - player2_deep_run: 1 if player2 has played 3+ matches in tournament, 0 otherwise
        """
        # Track matches played by each player in each tournament
        # Structure: {(player, tournament): match_count}
        tournament_matches = {}
        
        # Initialize result columns as lists
        p1_matches_in_tournament = []
        p2_matches_in_tournament = []
        tournament_exp_diff = []
        p1_deep_run = []
        p2_deep_run = []
        
        for idx, row in df.iterrows():
            player1 = row['player1']
            player2 = row['player2']
            tournament = row.get('tournament', 'Unknown')  # Get tournament name
            
            # Create keys for this tournament
            p1_key = (player1, tournament)
            p2_key = (player2, tournament)
            
            # Get current match count for each player in this tournament (before this match)
            p1_count = tournament_matches.get(p1_key, 0)
            p2_count = tournament_matches.get(p2_key, 0)
            
            # Calculate experience difference
            exp_diff = p1_count - p2_count
            
            # Deep run flags (3+ matches indicates deep tournament run)
            p1_is_deep = 1 if p1_count >= 3 else 0
            p2_is_deep = 1 if p2_count >= 3 else 0
            
            # Store results
            p1_matches_in_tournament.append(p1_count)
            p2_matches_in_tournament.append(p2_count)
            tournament_exp_diff.append(exp_diff)
            p1_deep_run.append(p1_is_deep)
            p2_deep_run.append(p2_is_deep)
            
            # Update match counts after this match
            tournament_matches[p1_key] = p1_count + 1
            tournament_matches[p2_key] = p2_count + 1
        
        # Add columns to dataframe
        df['player1_matches_in_tournament'] = p1_matches_in_tournament
        df['player2_matches_in_tournament'] = p2_matches_in_tournament
        df['tournament_experience_diff'] = tournament_exp_diff
        df['player1_deep_run'] = p1_deep_run
        df['player2_deep_run'] = p2_deep_run
        
        logger.info("Calculated tournament progression features: matches in tournament, deep run flags")
        
        return df
    
    def calculate_win_streaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate win/loss streak features for each player.
        
        Momentum is crucial in tennis - players on winning streaks often have psychological advantage.
        This tracks current winning/losing streaks overall and on specific surfaces.
        
        Args:
            df: DataFrame with match data (must be chronologically sorted)
            
        Returns:
            DataFrame with added win streak columns:
            - player1_win_streak: Current consecutive wins (negative for losses)
            - player2_win_streak: Current consecutive wins (negative for losses)
            - player1_surface_win_streak: Win streak on current surface
            - player2_surface_win_streak: Win streak on current surface
            - player1_wins_last_5: Wins in last 5 matches
            - player2_wins_last_5: Wins in last 5 matches
            - streak_advantage: Difference in overall win streaks
            - surface_streak_advantage: Difference in surface win streaks
        """
        # Track win/loss streaks for each player
        player_streak = {}  # {player: current_streak} (positive=wins, negative=losses)
        player_surface_streak = {}  # {(player, surface): current_streak}
        player_recent_results = {}  # {player: [list of recent results]}
        
        # Initialize result columns
        p1_win_streak = []
        p2_win_streak = []
        p1_surface_win_streak = []
        p2_surface_win_streak = []
        p1_wins_last_5 = []
        p2_wins_last_5 = []
        streak_advantage = []
        surface_streak_advantage = []
        
        for idx, row in df.iterrows():
            player1 = row['player1']
            player2 = row['player2']
            winner = row['winner']
            surface = row.get('surface', 'Unknown')
            
            # Initialize player trackers if new
            if player1 not in player_streak:
                player_streak[player1] = 0
                player_recent_results[player1] = []
            if player2 not in player_streak:
                player_streak[player2] = 0
                player_recent_results[player2] = []
            
            p1_surface_key = (player1, surface)
            p2_surface_key = (player2, surface)
            if p1_surface_key not in player_surface_streak:
                player_surface_streak[p1_surface_key] = 0
            if p2_surface_key not in player_surface_streak:
                player_surface_streak[p2_surface_key] = 0
            
            # Get current streaks BEFORE this match
            p1_streak = player_streak[player1]
            p2_streak = player_streak[player2]
            p1_surf_streak = player_surface_streak[p1_surface_key]
            p2_surf_streak = player_surface_streak[p2_surface_key]
            
            # Get wins in last 5 matches
            p1_recent = player_recent_results[player1]
            p2_recent = player_recent_results[player2]
            p1_last_5_wins = sum(1 for result in p1_recent[-5:] if result == 1)
            p2_last_5_wins = sum(1 for result in p2_recent[-5:] if result == 1)
            
            # Store stats for this match
            p1_win_streak.append(p1_streak)
            p2_win_streak.append(p2_streak)
            p1_surface_win_streak.append(p1_surf_streak)
            p2_surface_win_streak.append(p2_surf_streak)
            p1_wins_last_5.append(p1_last_5_wins)
            p2_wins_last_5.append(p2_last_5_wins)
            streak_advantage.append(p1_streak - p2_streak)
            surface_streak_advantage.append(p1_surf_streak - p2_surf_streak)
            
            # Update streaks AFTER this match
            if winner == player1:
                # Player1 won
                if p1_streak >= 0:
                    player_streak[player1] = p1_streak + 1  # Continue win streak
                else:
                    player_streak[player1] = 1  # Start new win streak
                
                if p1_surf_streak >= 0:
                    player_surface_streak[p1_surface_key] = p1_surf_streak + 1
                else:
                    player_surface_streak[p1_surface_key] = 1
                
                player_recent_results[player1].append(1)  # Win
                
                # Player2 lost
                if p2_streak <= 0:
                    player_streak[player2] = p2_streak - 1  # Continue loss streak
                else:
                    player_streak[player2] = -1  # Start new loss streak
                
                if p2_surf_streak <= 0:
                    player_surface_streak[p2_surface_key] = p2_surf_streak - 1
                else:
                    player_surface_streak[p2_surface_key] = -1
                
                player_recent_results[player2].append(0)  # Loss
                
            else:  # winner == player2
                # Player2 won
                if p2_streak >= 0:
                    player_streak[player2] = p2_streak + 1
                else:
                    player_streak[player2] = 1
                
                if p2_surf_streak >= 0:
                    player_surface_streak[p2_surface_key] = p2_surf_streak + 1
                else:
                    player_surface_streak[p2_surface_key] = 1
                
                player_recent_results[player2].append(1)  # Win
                
                # Player1 lost
                if p1_streak <= 0:
                    player_streak[player1] = p1_streak - 1
                else:
                    player_streak[player1] = -1
                
                if p1_surf_streak <= 0:
                    player_surface_streak[p1_surface_key] = p1_surf_streak - 1
                else:
                    player_surface_streak[p1_surface_key] = -1
                
                player_recent_results[player1].append(0)  # Loss
        
        # Add columns to dataframe
        df['player1_win_streak'] = p1_win_streak
        df['player2_win_streak'] = p2_win_streak
        df['player1_surface_win_streak'] = p1_surface_win_streak
        df['player2_surface_win_streak'] = p2_surface_win_streak
        df['player1_wins_last_5'] = p1_wins_last_5
        df['player2_wins_last_5'] = p2_wins_last_5
        df['streak_advantage'] = streak_advantage
        df['surface_streak_advantage'] = surface_streak_advantage
        
        logger.info("Calculated win streak features: streaks, surface streaks, recent form")
        
        return df
    
    def get_all_ratings(self) -> Dict[str, Dict[str, float]]:
        """Get current ratings for all players (general, surface-specific, and tournament-specific)."""
        return {
            'general': self.player_ratings.copy(),
            'hard': self.player_ratings_hard.copy(),
            'clay': self.player_ratings_clay.copy(),
            'grass': self.player_ratings_grass.copy(),
            'carpet': self.player_ratings_carpet.copy(),
            'grand_slam': self.player_ratings_grand_slam.copy(),
            'masters': self.player_ratings_masters.copy(),
            'atp500': self.player_ratings_atp500.copy(),
            'atp250': self.player_ratings_atp250.copy()
        }


class Calculator:
    def __init__(self, data):
        self.data = data

    def calculate_mean(self, column):
        return self.data[column].mean()

    def calculate_median(self, column):
        return self.data[column].median()

    def calculate_std_dev(self, column):
        return self.data[column].std()

    def calculate_win_percentage(self, team):
        total_matches = len(self.data)
        wins = len(self.data[self.data['winner'] == team])
        return (wins / total_matches) * 100 if total_matches > 0 else 0

    def calculate_statistics(self):
        stats = {
            'mean': self.calculate_mean('score'),
            'median': self.calculate_median('score'),
            'std_dev': self.calculate_std_dev('score'),
        }
        return stats