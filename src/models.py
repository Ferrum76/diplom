import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib


class PricePredictor:
    """Класс для построения и обновления модели прогнозирования цен."""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.model = None

    def preprocess_data(self, df):
        """Предобработка данных."""
        required_columns = ['count', 'add_cost']
        # Проверяем, какие столбцы присутствуют в данных
        available_columns = df.columns.tolist()
        subset_columns = [col for col in required_columns if col in available_columns]
        # Удаляем строки с пропущенными значениями в обязательных столбцах
        df = df.dropna(subset=subset_columns)
        categorical_cols = ['company', 'product']
        df = pd.get_dummies(df, columns=categorical_cols)
        return df

    def train_model(self):
        """Обучение модели на данных из базы данных."""
        print("Получение данных из базы данных для обучения модели...")
        df = self.db_manager.fetch_data()
        df_processed = self.preprocess_data(df)

        X = df_processed.drop(['price'], axis=1)
        y = df_processed['price']

        print("Разделение данных на обучающую и тестовую выборки...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("Обучение модели градиентного бустинга...")
        self.model = GradientBoostingRegressor()
        self.model.fit(X_train, y_train)

        print("Оценка модели на тестовой выборке...")
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Среднеквадратичная ошибка на тестовой выборке: {mse:.2f}")

        # Вывод важности признаков
        feature_importances = pd.Series(
            self.model.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
        print("Важность признаков:")
        print(feature_importances)

    def predict_price(self, input_data):
        """Прогнозирование цены на основе входных данных."""
        print("Предобработка введенных данных...")
        df_input = pd.DataFrame([input_data])
        df_input_processed = self.preprocess_data(df_input)

        # Обеспечение соответствия столбцов с обучающими данными
        df_train = self.db_manager.fetch_data()
        df_train_processed = self.preprocess_data(df_train)
        df_train_features = df_train_processed.drop('price', axis=1)

        # Добавляем отсутствующие столбцы в df_input_processed
        missing_cols = set(df_train_features.columns) - set(df_input_processed.columns)
        for col in missing_cols:
            df_input_processed[col] = 0
        # Убираем лишние столбцы, которые отсутствуют в обучающих данных
        df_input_processed = df_input_processed[df_train_features.columns]

        print("Выполнение прогноза цены...")
        prediction = self.model.predict(df_input_processed)
        return prediction[0]

    def update_model(self):
        """Обновление модели с новыми данными."""
        self.train_model()
        joblib.dump(self.model, 'price_model.pkl')
        print("Модель сохранена в 'price_model.pkl'.")

    def load_model(self):
        """Загрузка модели из файла."""
        self.model = joblib.load('price_model.pkl')
        print("Модель загружена из 'price_model.pkl'.")
