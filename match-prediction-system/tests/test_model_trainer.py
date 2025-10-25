import unittest
from src.model_trainer.train import Trainer
from src.model_trainer.model import Model
from src.data_processor.data_loader import DataLoader

class TestModelTrainer(unittest.TestCase):

    def setUp(self):
        self.data_loader = DataLoader()
        self.trainer = Trainer(Model())

    def test_training_process(self):
        # Load sample data
        data = self.data_loader.load_data('path/to/sample/data.csv')
        self.trainer.train(data)
        self.assertTrue(self.trainer.model.is_trained)

    def test_model_saving(self):
        self.trainer.save_model('models/saved_models/test_model.pkl')
        self.assertTrue(os.path.exists('models/saved_models/test_model.pkl'))

if __name__ == '__main__':
    unittest.main()