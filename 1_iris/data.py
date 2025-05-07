from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import os

file_path = 'dataset/iris.csv'

def load_data(file_path: str) -> str:
    """Загрузка датасета Iris и сохранение его в заданный файл."""
    
    # Загружаем датасет Iris
    iris = load_iris()
    
    # Преобразуем данные в DataFrame
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    
    # Создаем директорию, если её нет
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Сохраняем в CSV
    iris_df.to_csv(file_path, index=False)
    
    return file_path


def prepare_data(csv_path: str) -> list[str]:
    """Чтение загруженного датасета и разделение на train и test выборки."""
    
    # Загружаем данные из CSV
    iris_df = pd.read_csv(csv_path)
    
    # Разделяем данные на признаки (X) и целевую переменную (y)
    X = iris_df.drop(columns=['target'])
    y = iris_df['target']
    
    # Разделяем на обучающую и тестовую выборки в соотношении 80%/20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Объединяем X и y для обучающей выборки
    iris_train = pd.concat([X_train, y_train], axis=1)
    iris_test = pd.concat([X_test, y_test], axis=1)
    
    # Сохраняем в CSV
    iris_train.to_csv('dataset/iris_train.csv', index=False)
    iris_test.to_csv('dataset/iris_test.csv', index=False)
    
    return ['dataset/iris_train.csv', 'dataset/iris_test.csv']


load_data(file_path)
prepare_data(file_path)