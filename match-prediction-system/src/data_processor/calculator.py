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
        Form is calculated as win percentage over the last N matches.
        
        Args:
            df: DataFrame with match data (must be chronologically sorted)
            window: Number of recent matches to consider for form calculation
            
        Returns:
            DataFrame with added form columns
        """
        # Initialize form columns as lists (faster than df.at[])
        player1_form = []
        player2_form = []
        player1_form_wins = []
        player2_form_wins = []
        player1_form_matches = []
        player2_form_matches = []
        
        # Track match history for each player using deque for efficiency
        from collections import deque
        player_match_history = {}  # {player: deque of is_win booleans}
        
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
            p1_history = player_match_history[player1]
            p2_history = player_match_history[player2]
            
            # Player 1 form
            if len(p1_history) > 0:
                p1_wins = sum(p1_history)
                p1_matches = len(p1_history)
                player1_form.append((p1_wins / p1_matches) * 100)
                player1_form_wins.append(p1_wins)
                player1_form_matches.append(p1_matches)
            else:
                player1_form.append(0.0)
                player1_form_wins.append(0)
                player1_form_matches.append(0)
            
            # Player 2 form
            if len(p2_history) > 0:
                p2_wins = sum(p2_history)
                p2_matches = len(p2_history)
                player2_form.append((p2_wins / p2_matches) * 100)
                player2_form_wins.append(p2_wins)
                player2_form_matches.append(p2_matches)
            else:
                player2_form.append(0.0)
                player2_form_wins.append(0)
                player2_form_matches.append(0)
            
            # Update match history after this match (deque auto-limits to window size)
            player_match_history[player1].append(winner == player1)
            player_match_history[player2].append(winner == player2)
        
        # Add columns to dataframe (much faster than df.at[])
        df['player1_form'] = player1_form
        df['player2_form'] = player2_form
        df['player1_form_wins'] = player1_form_wins
        df['player2_form_wins'] = player2_form_wins
        df['player1_form_matches'] = player1_form_matches
        df['player2_form_matches'] = player2_form_matches
        
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