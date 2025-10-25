class Evaluator:
    def __init__(self, model, validation_data):
        self.model = model
        self.validation_data = validation_data

    def evaluate(self):
        predictions = self.model.predict(self.validation_data.features)
        accuracy = self.calculate_accuracy(predictions, self.validation_data.labels)
        return {
            'accuracy': accuracy,
            'predictions': predictions
        }

    def calculate_accuracy(self, predictions, labels):
        correct_predictions = sum(pred == label for pred, label in zip(predictions, labels))
        return correct_predictions / len(labels) if labels else 0.0