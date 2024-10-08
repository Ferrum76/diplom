from src.database import DatabaseManager
from src.models import PricePredictor
import pandas as pd


def main():
    print("Инициализация менеджера базы данных...")
    db_manager = DatabaseManager()

    print("Загрузка данных из 'csv_data.csv'...")
    data = pd.read_csv('csv_data.csv')

    print("Вставка данных в базу данных...")
    db_manager.insert_product_data(data)

    print("Инициализация модели прогнозирования цен...")
    predictor = PricePredictor(db_manager)

    print("Обучение модели на загруженных данных...")
    predictor.train_model()
    predictor.update_model()

    print("\n=== Обучение завершено ===\n")

    print("Введите данные для прогнозирования цены:")
    count = int(input("Количество продаж (count): "))
    add_cost = float(input("Затраты на продвижение (add_cost): "))
    company = input("Компания-производитель (company): ").lower()
    product = input("Наименование продукта (product): ").lower()

    new_data = {
        'count': count,
        'add_cost': add_cost,
        'company': company,
        'product': product
    }

    print("\nАнализ введенных данных и прогнозирование цены...")
    predicted_price = predictor.predict_price(new_data)
    print(f"\nПрогнозируемая цена для продукта '{product}' "
          f"от компании '{company}': {predicted_price:.2f}")

    # Закрытие соединения с базой данных
    db_manager.close()
    print("\nПрограмма завершена.")


if __name__ == '__main__':
    main()
