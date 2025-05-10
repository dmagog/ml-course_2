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
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("LogisticRegressionExperiment")
    with mlflow.start_run():
        model = LogisticRegression(
            max_iter=config_lr["logistic_regression"]["max_iter"],
            C=config_lr["logistic_regression"].get("C", 1.0)
        )
        data = get_data_lr()
        train_lr(model, data["x_train"], data["y_train"])
        acc, f1, cm, auc = test_lr(model, data["x_test"], data["y_test"])

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc if auc is not None else 0.0)
        mlflow.log_dict({"confusion_matrix": cm.tolist()}, "confusion_matrix.json")
        
        mlflow.sklearn.log_model(model, "model")


def run_decision_tree():
    import mlflow
    import mlflow.sklearn
    
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("DecisionTreeExperiment")
    with mlflow.start_run():
        model = DecisionTreeClassifier(
            random_state=config_dt["random_state"],
            max_depth=config_dt["decision_tree"]["max_depth"],
            criterion=config_dt["decision_tree"].get("criterion", "gini")
        )
        data = get_data_dt()
        train_dt(model, data["x_train"], data["y_train"])
        acc, f1, cm, auc = test_dt(model, data["x_test"], data["y_test"])

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc if auc is not None else 0.0)
        mlflow.log_dict({"confusion_matrix": cm.tolist()}, "confusion_matrix.json")

        mlflow.sklearn.log_model(model, "model")



with DAG(
    dag_id="ml_model_mlflow",
    default_args=default_args,
    description="Train ML models and log to MLflow",
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