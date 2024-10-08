import sqlite3
import pandas as pd


class DatabaseManager:
    """Класс для управления операциями с базой данных."""

    def __init__(self, db_name='data.db'):
        self.conn = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        """Создание таблицы в базе данных."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS product_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                price REAL,
                count INTEGER,
                add_cost REAL,
                company TEXT,
                product TEXT
            )
        ''')
        self.conn.commit()

    def insert_product_data(self, df):
        """Вставка данных о продуктах в базу данных."""
        df.to_sql('product_data', self.conn, if_exists='append', index=False)

    def fetch_data(self):
        """Получение данных из базы данных."""
        query = '''
            SELECT * FROM product_data
        '''
        return pd.read_sql_query(query, self.conn)

    def close(self):
        """Закрытие соединения с базой данных."""
        self.conn.close()
