class PredictionService:
    def __init__(self, model, data_processor):
        self.model = model
        self.data_processor = data_processor

    def predict_match_outcome(self, match_data):
        processed_data = self.data_processor.process(match_data)
        prediction = self.model.predict(processed_data)
        return prediction

    def get_prediction_percentage(self, match_data):
        processed_data = self.data_processor.process(match_data)
        probabilities = self.model.predict_proba(processed_data)
        return {
            "team_a_win": probabilities[0],
            "draw": probabilities[1],
            "team_b_win": probabilities[2]
        }