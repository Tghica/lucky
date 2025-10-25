from src.predictor.prediction_service import PredictionService
from src.predictor.match_analyzer import MatchAnalyzer

def test_prediction_service():
    prediction_service = PredictionService()
    match_data = {
        'team_a': 'Team A',
        'team_b': 'Team B',
        'previous_results': [1, 0, 1, 0, 1],  # Example previous results
        'current_form': [1, 1, 0, 1],  # Example current form
    }
    
    prediction = prediction_service.predict(match_data)
    assert 'winning_probability' in prediction
    assert isinstance(prediction['winning_probability'], float)

def test_match_analyzer():
    match_analyzer = MatchAnalyzer()
    match_data = {
        'team_a': 'Team A',
        'team_b': 'Team B',
        'previous_results': [1, 0, 1, 0, 1],
        'current_form': [1, 1, 0, 1],
    }
    
    analysis = match_analyzer.analyze(match_data)
    assert 'winning_probability' in analysis
    assert 0 <= analysis['winning_probability'] <= 1  # Probability should be between 0 and 1