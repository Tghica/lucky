class Trainer:
    def __init__(self, model, data_loader, evaluator):
        self.model = model
        self.data_loader = data_loader
        self.evaluator = evaluator

    def train(self, epochs, learning_rate):
        for epoch in range(epochs):
            data = self.data_loader.load_training_data()
            self.model.train(data, learning_rate)
            validation_data = self.data_loader.load_validation_data()
            metrics = self.evaluator.evaluate(self.model, validation_data)
            print(f'Epoch {epoch + 1}/{epochs}, Metrics: {metrics}')

    def save_model(self, file_path):
        self.model.save(file_path)