"""Data fetching utilities for tennis match data (RapidAPI Tennis API)"""

from .fetch_rapidapi import fetch_tennis_matches, RapidAPITennisClient

__all__ = [
    'fetch_tennis_matches',
    'RapidAPITennisClient',
]