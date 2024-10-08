import sqlite3

import pandas as pd


class DatabaseManager:
    """
    Класс для управления операциями с базой данных SQLite.

    Атрибуты:
        conn (sqlite3.Connection): Объект соединения с базой данных.

    Методы:
        create_tables(): Создает необходимые таблицы в базе данных.
        insert_product_data(df): Вставляет данные о продуктах в базу данных.
        fetch_data(): Получает данные из базы данных.
        close(): Закрывает соединение с базой данных.
    """

    def __init__(self, db_name='data.db'):
        """
        Инициализирует соединение с базой данных и создает таблицы.

        Args:
            db_name (str): Имя файла базы данных. По умолчанию 'data.db'.
        """
        self.conn = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        """
        Создает таблицу 'product_data' в базе данных, если она не существует.
        """
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
        """
        Вставляет данные о продуктах из DataFrame в таблицу 'product_data'.

        Args:
            df (pd.DataFrame): Данные для вставки.
        """
        df.to_sql('product_data', self.conn, if_exists='append', index=False)

    def fetch_data(self):
        """
        Получает все данные из таблицы 'product_data'.

        Returns:
            pd.DataFrame: Данные из базы данных.
        """
        query = '''
            SELECT * FROM product_data
        '''
        return pd.read_sql_query(query, self.conn)

    def close(self):
        """
        Закрывает соединение с базой данных.
        """
        self.conn.close()
