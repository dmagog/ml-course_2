from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json


model_file_path = 'model/model.pkl'
file_path = 'dataset/iris.csv'
file_path_train = 'dataset/iris_train.csv'
file_path_test = 'dataset/iris_test.csv'

def load_data(file_path: str):
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


def prepare_data(csv_path: str):
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





def train(train_csv: str):
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




# Основной код для DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 5, 6),
    'retries': 1,
    'email_on_failure': False,
    'email_on_retry': False,
    'email_on_success': False,
    'depends_on_past': False,
}

dag = DAG(
    'model_training_dag',
    default_args=default_args,
    description='DAG for training and evaluating a model for IRIS dataset',
    schedule_interval=None,  # Мы можем запускать DAG вручную
    catchup=False,
)

# Задачи

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    op_kwargs={
            "file_path": file_path
        },
    provide_context=True,
    dag=dag,
)

prepare_data_task = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data,
    op_kwargs={
            "csv_path": file_path
        },
    provide_context=True,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train',
    python_callable=train,
    op_kwargs={
            "train_csv": file_path_train
        },
    provide_context=True,
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='test',
    python_callable=test,
    op_kwargs={
            "model_path": model_file_path,
            "test_csv": file_path_test
        },
    provide_context=True,
    dag=dag,
)

# Задачи выполняются последовательно
load_data_task >> prepare_data_task >> train_model_task >> evaluate_model_task