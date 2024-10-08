import unittest
import pandas as pd
from src.database import DatabaseManager
from src.models import PricePredictor
import joblib
import os


class TestPricePredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.db_manager = DatabaseManager(':memory:')
        # Пример данных
        cls.data = pd.DataFrame({
            'price': [1000, 2000, 1500],
            'count': [100, 150, 120],
            'add_cost': [300, 400, 350],
            'company': ['CompanyA', 'CompanyB', 'CompanyA'],
            'product': ['ProductX', 'ProductY', 'ProductZ']
        })

        # Приводим названия к нижнему регистру
        cls.data['company'] = cls.data['company'].str.lower()
        cls.data['product'] = cls.data['product'].str.lower()

        cls.db_manager.insert_product_data(cls.data)
        cls.predictor = PricePredictor(cls.db_manager)
        cls.predictor.train_model()

    @classmethod
    def tearDownClass(cls):
        cls.db_manager.close()

    def test_preprocess_data(self):
        df_processed = self.predictor.preprocess_data(self.data)
        self.assertIn('company_companya', df_processed.columns)
        self.assertIn('product_productx', df_processed.columns)

    def test_predict_price(self):
        input_data = {
            'count': 130,
            'add_cost': 320,
            'company': 'companya',
            'product': 'productx'
        }
        price = self.predictor.predict_price(input_data)
        self.assertIsInstance(price, float)

    def test_update_model(self):
        # Проверяем, что модель сохраняется без ошибок
        self.predictor.update_model()
        # Сохраняем модель в отдельный файл для тестирования
        test_model_path = 'test_price_model.pkl'
        joblib.dump(self.predictor.model, test_model_path)
        self.assertTrue(os.path.exists(test_model_path))
        # Удаляем тестовую модель после проверки
        os.remove(test_model_path)

    def test_load_model(self):
        # Проверяем, что модель загружается без ошибок
        # Сохраняем модель в отдельный файл для тестирования
        test_model_path = 'test_price_model.pkl'
        joblib.dump(self.predictor.model, test_model_path)
        # Загружаем модель
        self.predictor.model = None  # Сбрасываем текущую модель
        self.predictor.model = joblib.load(test_model_path)
        self.assertIsNotNone(self.predictor.model)
        # Удаляем тестовую модель после проверки
        os.remove(test_model_path)


if __name__ == '__main__':
    unittest.main()
