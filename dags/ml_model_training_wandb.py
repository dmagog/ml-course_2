import json
import numpy as np
import matplotlib.pyplot as plt
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

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
    import wandb
    import joblib

    # конфигурационный словарь
    config = {
        "learning_rate": 0.01,
        "batch_size": 32,
        "num_epochs": 10,
    }

    wandb.init(project='ML Models', name='Logistic Regression')
    wandb.config.update(config) # передаем конфиг в ClearML


    wandb.config.update(config_lr)
    model = LogisticRegression(
        max_iter=config_lr["logistic_regression"]["max_iter"],
        C=config_lr["logistic_regression"].get("C", 1.0)
    )
    data = get_data_lr()
    train_lr(model, data["x_train"], data["y_train"])
    acc, f1, cm, auc = test_lr(model, data["x_test"], data["y_test"])

    wandb.log({
        "accuracy": acc,
        "f1_score": f1,
        "auc_roc": auc
    })  

    joblib.dump(model, "model.pkl")
    model_artifact = wandb.Artifact("trained_model", type="model")
    model_artifact.add_file("model.pkl")
    wandb.log_artifact(model_artifact)
    wandb.save('model.pkl')
    wandb.finish()


def run_decision_tree():
    import wandb
    import joblib

    wandb.init(project='ML Models', name='Decision Tree')
    wandb.config.update(config_dt)
    model = DecisionTreeClassifier(
        random_state=config_dt["random_state"],
        max_depth=config_dt["decision_tree"]["max_depth"],
        criterion=config_dt["decision_tree"].get("criterion", "gini")
    )
    data = get_data_dt()
    train_dt(model, data["x_train"], data["y_train"])
    acc, f1, cm, auc = test_dt(model, data["x_test"], data["y_test"])

    wandb.log({
        "accuracy": acc,
        "f1_score": f1,
        "auc_roc": auc
    })  

    joblib.dump(model, "model.pkl")
    model_artifact = wandb.Artifact("trained_model", type="model")
    model_artifact.add_file("model.pkl")
    wandb.log_artifact(model_artifact)
    wandb.save('model.pkl')
    wandb.finish()


with DAG(
    dag_id="ml_model_wandb",
    default_args=default_args,
    description="Train ML models and log to W&B",
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