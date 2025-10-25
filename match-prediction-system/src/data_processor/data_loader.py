import pandas as pd
import os
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and merges multiple Excel/CSV sheets into a single dataset."""
    
    def __init__(self, sheets_folder: str = "data/sheets"):
        self.sheets_folder = Path(sheets_folder)
        self.required_columns = [
            'date',
            'player1',
            'player2', 
            'winner',
            'stadium',
            'surface',  # type of field
            'description',
            'nation'
        ]
    
    def load_sheet(self, file_path: Path) -> pd.DataFrame:
        """Load a single sheet (Excel or CSV)."""
        try:
            if file_path.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_path}")
                return None
            
            logger.info(f"Loaded {len(df)} rows from {file_path.name}")
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different sheets."""
        # Create a mapping for common variations
        column_mapping = {
            # Date variations
            'Date': 'date',
            'match_date': 'date',
            'Match Date': 'date',
            
            # Player names
            'Player_1': 'player1',
            'Player 1': 'player1',
            'player_1': 'player1',
            'Player1': 'player1',
            'winner_name': 'player1',
            
            'Player_2': 'player2',
            'Player 2': 'player2',
            'player_2': 'player2',
            'Player2': 'player2',
            'loser_name': 'player2',
            
            # Winner
            'Winner': 'winner',
            'winner_name': 'winner',
            
            # Stadium/Venue
            'Stadium': 'stadium',
            'venue': 'stadium',
            'Venue': 'stadium',
            'Court': 'stadium',
            
            # Surface type
            'Surface': 'surface',
            'court_surface': 'surface',
            'Court Surface': 'surface',
            'field_type': 'surface',
            
            # Description (Tournament name)
            'Description': 'description',
            'match_description': 'description',
            'tournament': 'description',
            'Tournament': 'description',
            'tourney_name': 'description',
            
            # Nation/Country/Location
            'Nation': 'nation',
            'Country': 'nation',
            'country': 'nation',
            'location': 'nation',
            'tourney_location': 'nation'
        }
        
        # Rename columns based on mapping
        df = df.rename(columns=column_mapping)
        
        # Convert to lowercase for consistency
        df.columns = df.columns.str.lower()
        
        return df
    
    def merge_all_sheets(self, output_path: str = "data/processed/match_info.csv") -> pd.DataFrame:
        """Merge all sheets from the folder into a single DataFrame."""
        all_dataframes = []
        
        # Get all files in the sheets folder
        sheet_files = list(self.sheets_folder.glob('*'))
        
        if not sheet_files:
            logger.error(f"No files found in {self.sheets_folder}")
            return None
        
        logger.info(f"Found {len(sheet_files)} files to process")
        
        for file_path in sheet_files:
            df = self.load_sheet(file_path)
            if df is not None:
                df = self.standardize_columns(df)
                all_dataframes.append(df)
        
        if not all_dataframes:
            logger.error("No data loaded from any files")
            return None
        
        # Merge all dataframes
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        logger.info(f"Total rows after merging: {len(merged_df)}")
        
        # Check which required columns are present
        missing_columns = set(self.required_columns) - set(merged_df.columns)
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
        
        # Select only the required columns that exist
        available_columns = [col for col in self.required_columns if col in merged_df.columns]
        merged_df = merged_df[available_columns]
        
        # Remove duplicates
        original_count = len(merged_df)
        merged_df = merged_df.drop_duplicates()
        logger.info(f"Removed {original_count - len(merged_df)} duplicate rows")
        
        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Merged data saved to {output_path}")
        
        return merged_df
    
    def create_player_database(self, match_df: pd.DataFrame, output_path: str = "data/processed/player_info.csv") -> pd.DataFrame:
        """Create a comprehensive player database from match data and ATP players file."""
        logger.info("Creating player database...")
        
        # Try to load the ATP players file
        players_file = Path("data/sheets/archive 2/atp_players.csv")
        atp_players_df = None
        
        if players_file.exists():
            try:
                atp_players_df = pd.read_csv(players_file)
                logger.info(f"Loaded {len(atp_players_df)} players from ATP players database")
                
                # Standardize player names to match format in matches (Last F.)
                atp_players_df['full_name'] = atp_players_df['name_last'] + ' ' + atp_players_df['name_first'].str[0] + '.'
                atp_players_df['player_name'] = atp_players_df['full_name']
            except Exception as e:
                logger.warning(f"Could not load ATP players file: {e}")
        
        # Get all unique players from match data
        match_players = set(match_df['player1'].unique().tolist() + match_df['player2'].unique().tolist())
        
        player_stats = []
        
        for player in match_players:
            # Get all matches where this player participated
            player_matches = match_df[(match_df['player1'] == player) | (match_df['player2'] == player)]
            
            # Calculate statistics
            total_matches = len(player_matches)
            wins = len(player_matches[player_matches['winner'] == player])
            losses = total_matches - wins
            win_rate = (wins / total_matches * 100) if total_matches > 0 else 0
            
            # Surface-specific stats
            surface_stats = {}
            for surface in match_df['surface'].unique():
                surface_matches = player_matches[player_matches['surface'] == surface]
                surface_wins = len(surface_matches[surface_matches['winner'] == player])
                surface_total = len(surface_matches)
                surface_stats[f'{surface.lower()}_matches'] = surface_total
                surface_stats[f'{surface.lower()}_wins'] = surface_wins
                surface_stats[f'{surface.lower()}_win_rate'] = (surface_wins / surface_total * 100) if surface_total > 0 else 0
            
            # Date range
            first_match = player_matches['date'].min()
            last_match = player_matches['date'].max()
            
            # Most played tournaments
            tournament_counts = player_matches['description'].value_counts()
            most_played_tournament = tournament_counts.index[0] if len(tournament_counts) > 0 else None
            
            # Get additional info from ATP players database if available
            hand = None
            dob = None
            ioc = None
            height = None
            player_id = None
            
            if atp_players_df is not None:
                player_info_row = atp_players_df[atp_players_df['player_name'] == player]
                if not player_info_row.empty:
                    hand = player_info_row.iloc[0]['hand'] if 'hand' in player_info_row.columns else None
                    dob = player_info_row.iloc[0]['dob'] if 'dob' in player_info_row.columns else None
                    ioc = player_info_row.iloc[0]['ioc'] if 'ioc' in player_info_row.columns else None
                    height = player_info_row.iloc[0]['height'] if 'height' in player_info_row.columns else None
                    player_id = player_info_row.iloc[0]['player_id'] if 'player_id' in player_info_row.columns else None
            
            player_info = {
                'player_id': player_id,
                'player_name': player,
                'hand': hand,
                'dob': dob,
                'country': ioc,
                'height': height,
                'total_matches': total_matches,
                'total_wins': wins,
                'total_losses': losses,
                'win_rate': round(win_rate, 2),
                'first_match_date': first_match,
                'last_match_date': last_match,
                'most_played_tournament': most_played_tournament,
                **surface_stats
            }
            
            player_stats.append(player_info)
        
        # Create DataFrame
        player_df = pd.DataFrame(player_stats)
        
        # Sort by total matches (most active players first)
        player_df = player_df.sort_values('total_matches', ascending=False).reset_index(drop=True)
        
        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        player_df.to_csv(output_path, index=False)
        logger.info(f"Player database saved to {output_path} with {len(player_df)} players")
        
        return player_df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics of the merged data."""
        summary = {
            'total_matches': len(df),
            'date_range': (df['date'].min(), df['date'].max()) if 'date' in df.columns else None,
            'unique_players': len(set(df['player1'].tolist() + df['player2'].tolist())) if 'player1' in df.columns else None,
            'surfaces': df['surface'].value_counts().to_dict() if 'surface' in df.columns else None,
            'nations': df['nation'].value_counts().to_dict() if 'nation' in df.columns else None,
            'missing_data': df.isnull().sum().to_dict()
        }
        return summary


if __name__ == "__main__":
    loader = DataLoader()
    merged_df = loader.merge_all_sheets()
    
    if merged_df is not None:
        summary = loader.get_data_summary(merged_df)
        print("\n=== Data Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")