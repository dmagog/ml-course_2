from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import json

from model1 import train as train_lr, test as test_lr
from model2 import train as train_dt, test as test_dt
from model1 import LogisticRegression, config as config_lr, get_data as get_data_lr
from model2 import DecisionTreeClassifier, config as config_dt, get_data as get_data_dt

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

def run_logistic_regression():
    from clearml import Task

    # конфигурационный словарь
    config = {
        "learning_rate": 0.01,
        "batch_size": 32,
        "num_epochs": 10,
    }

    task = Task.init(project_name='ML Models', task_name='Logistic Regression')
    logger = task.get_logger()

    task.connect(config) # передаем конфиг в ClearML

    task.connect(config_lr)
    model = LogisticRegression(
        max_iter=config_lr["logistic_regression"]["max_iter"],
        C=config_lr["logistic_regression"].get("C", 1.0)
    )
    data = get_data_lr()
    train_lr(model, data["x_train"], data["y_train"])
    acc, f1, cm, auc = test_lr(model, data["x_test"], data["y_test"])

    logger.report_scalar("accuracy", "value", acc, iteration=0)
    logger.report_scalar("f1_score", "value", f1, iteration=0)
    logger.report_scalar("auc_roc", "value", auc if auc is not None else 0.0, iteration=0)

    # Сохранение confusion matrix как JSON
    with open("confusion_matrix.json", "w") as f:
        json.dump(cm.tolist(), f)
    task.upload_artifact("confusion_matrix", "confusion_matrix.json")

    task.upload_artifact('model', model)
    task.close()


def run_decision_tree():
    from clearml import Task

    task = Task.init(project_name='ML Models', task_name='Decision Tree')
    logger = task.get_logger()
    
    task.connect(config_dt)
    model = DecisionTreeClassifier(
        random_state=config_dt["random_state"],
        max_depth=config_dt["decision_tree"]["max_depth"],
        criterion=config_dt["decision_tree"].get("criterion", "gini")
    )
    data = get_data_dt()
    train_dt(model, data["x_train"], data["y_train"])
    acc, f1, cm, auc = test_dt(model, data["x_test"], data["y_test"])

    logger.report_scalar("accuracy", "value", acc, iteration=0)
    logger.report_scalar("f1_score", "value", f1, iteration=0)
    logger.report_scalar("auc_roc", "value", auc if auc is not None else 0.0, iteration=0)

    # Сохранение confusion matrix как JSON
    with open("confusion_matrix.json", "w") as f:
        json.dump(cm.tolist(), f)
    task.upload_artifact("confusion_matrix", "confusion_matrix.json")

    task.upload_artifact('model', model)
    task.close()


with DAG(
    dag_id="ml_model_clearml",
    default_args=default_args,
    description="Train ML models and log to ClearML",
    schedule_interval=None,
    catchup=False,
    tags=["ml", "training"],
) as dag:

    train_logistic_regression = PythonOperator(
        task_id="train_logistic_regression",
        python_callable=run_logistic_regression,
    )

    train_decision_tree = PythonOperator(
        task_id="train_decision_tree",
        python_callable=run_decision_tree,
    )

    train_logistic_regression >> train_decision_tree