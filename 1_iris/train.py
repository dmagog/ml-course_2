import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def train(train_csv: str) -> str:
    """Обучение модели логистической регрессии на тренировочной выборке и сохранение модели."""
    
    # Загружаем данные из CSV
    train_df = pd.read_csv(train_csv)
    
    # Разделяем данные на признаки (X) и целевую переменную (y)
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    
    # Создаем модель логистической регрессии
    model = LogisticRegression(max_iter=200)
    
    # Обучаем модель
    model.fit(X_train, y_train)
    
    # Сохраняем модель в файл
    model_file_path = 'model/model.pkl'
    
    # Создаем директорию, если её нет
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    
    # Сохраняем модель с помощью joblib
    joblib.dump(model, model_file_path)
    
    return model_file_path

file_path = 'dataset/iris_train.csv'
train(file_path)