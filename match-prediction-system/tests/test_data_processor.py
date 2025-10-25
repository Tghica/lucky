import unittest
from src.data_processor.calculator import Calculator
from src.data_processor.data_loader import DataLoader
from src.data_processor.feature_engineering import FeatureEngineering

class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        self.data_loader = DataLoader()
        self.calculator = Calculator()
        self.feature_engineering = FeatureEngineering()

    def test_load_data(self):
        data = self.data_loader.load('path/to/raw/data.csv')
        self.assertIsNotNone(data)

    def test_calculate_statistics(self):
        data = self.data_loader.load('path/to/raw/data.csv')
        stats = self.calculator.calculate_statistics(data)
        self.assertIn('mean', stats)
        self.assertIn('median', stats)

    def test_feature_engineering(self):
        data = self.data_loader.load('path/to/raw/data.csv')
        features = self.feature_engineering.create_features(data)
        self.assertGreater(len(features), 0)

if __name__ == '__main__':
    unittest.main()