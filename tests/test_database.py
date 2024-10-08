import unittest
import pandas
from src.database import DatabaseManager


class TestDatabaseManager(unittest.TestCase):

    def setUp(self):
        # Используем базу данных в памяти для тестирования
        self.db_manager = DatabaseManager(':memory:')
        # Пример данных
        self.data = pandas.DataFrame({
            'price': [1000, 2000],
            'count': [100, 150],
            'add_cost': [300, 400],
            'company': ['CompanyA', 'CompanyB'],
            'product': ['ProductX', 'ProductY']
        })

    def tearDown(self):
        self.db_manager.close()

    def test_insert_and_fetch_data(self):
        # Вставка данных в базу данных
        self.db_manager.insert_product_data(self.data)
        # Получение данных из базы данных
        fetched_data = self.db_manager.fetch_data()
        # Проверка, что данные получены корректно
        self.assertEqual(len(fetched_data), 2)
        self.assertListEqual(fetched_data['price'].tolist(), [1000, 2000])

    def test_create_tables(self):
        # Проверка, что таблица создана
        cursor = self.db_manager.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='product_data'")
        table = cursor.fetchone()
        self.assertIsNotNone(table)
        self.assertEqual(table[0], 'product_data')


if __name__ == '__main__':
    unittest.main()
