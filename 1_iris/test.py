import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
import json

def test(model_path: str, test_csv: str) -> str:
    """Тестирование модели на тестовой выборке и сохранение результатов."""
    
    # Загружаем модель из файла
    model = joblib.load(model_path)
    
    # Загружаем данные из тестового CSV
    test_df = pd.read_csv(test_csv)
    
    # Разделяем данные на признаки (X) и целевую переменную (y)
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']
    
    # Выполняем предсказания
    y_pred = model.predict(X_test)
    
    # Вычисляем accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Генерируем отчет
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Создаем метрики
    metrics = {'accuracy': accuracy, 'report': report}
    
    # Сохраняем метрики в JSON файл
    metrics_file_path = 'model_metrics.json'
    with open(metrics_file_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics_file_path


model_file_path = 'model/model.pkl'
file_path = 'dataset/iris_test.csv'
test(model_file_path, file_path)