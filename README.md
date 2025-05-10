> *Внутри этого же проекта лежат DAG'и из урока 3 — про ирисы и погоду. Запуск я осуществляю внутри докера, поэтому делать отдельные контейнеры под отдельные задачи очень не хочется, пока они не кофликтуют внутри себя. А они не конфликтуют.*


# ML Model Training with Airflow

Этот проект реализует пайплайн машинного обучения с использованием Apache Airflow для оркестрации, и логирования экспериментов через:
- ✅ [MLflow](https://mlflow.org/)
- ✅ [ClearML](https://clear.ml/)
- ✅ [Weights & Biases (W&B)](https://wandb.ai/)

*MLflow* поднят локально внутри того же docker-контейнера.
---

## 📁 Структура проекта

```
.
├── dags/
│   ├── ml_model_training_mlflow.py         # DAG для MLflow
│   ├── ml_model_training_clearml.py        # DAG для ClearML
│   ├── ml_model_training_wandb.py          # DAG для Weights & Biases
│   └── ...
├── model1.py                                # Логистическая регрессия (универсальный модуль)
├── model2.py                                # Решающее дерево (универсальный модуль)
├── data.py                                  # Загрузка и разделение данных
├── clearml.conf                             # Конфигурация для ClearML
├── docker-compose.yml                       # Docker-окружение
```

---

## 🚀 Запуск

### 1. Подготовка окружения

Убедитесь, что у вас установлен [Docker](https://www.docker.com/) и [Docker Compose](https://docs.docker.com/compose/).

### 2. Настройка API-ключей

#### MLflow
Локальный, не требует ключей.

#### ClearML
Укажите файл `clearml.conf` и примонтируйте в Docker:
```yaml
volumes:
  - ./clearml.conf:/home/airflow/.clearml.conf
```

#### Weights & Biases
Установите переменную:
```env
WANDB_API_KEY=your-api-key
```

### 3. Запуск

```bash
docker-compose up --build
```

Откройте Airflow UI на `http://localhost:8081`

---

## ⚙️ Функции

- Модульная структура моделей: код `train()` и `test()` переиспользуется в разных DAG
- Централизованное логирование метрик, параметров и артефактов
- Конфигурируемое логирование: выбор между MLflow, ClearML, W&B

---

## 📊 Логирование

| Логируется       | MLflow | ClearML | W&B   |
|------------------|--------|---------|--------|
| Accuracy         | ✅      | ✅       | ✅     |
| F1 Score         | ✅      | ✅       | ✅     |
| AUC-ROC          | ✅      | ✅       | ✅     |
| Confusion Matrix | ✅      | ✅       | ✅     |
| Модель           | ✅      | ✅       | ✅     |

---

## 📌 Заметки

- Метрики логируются в DAG после вызова `test()`, чтобы обеспечить универсальность моделей
- Артефакты сохраняются как `.json` и `.pkl` 
- Все DAG используют `PythonOperator`


## 📷 Примеры логов и артефактов

### MLFlow

![MLFlow Example](/docs/mlflow_example_1.png)
![MLFlow Example](/docs/mlflow_example_2.png)
![MLFlow Example](/docs/mlflow_example_3.png)

### ClearML

![ClearML Example](/docs/clearml_example_1.png)
![ClearML Example](/docs/clearml_example_2.png)
![ClearML Example](/docs/clearml_example_3.png)
![ClearML Example](/docs/clearml_example_4.png)
![ClearML Example](/docs/clearml_example_5.png)


### Weights & Biases (W&B)

![WandB Run](/docs/wandb_example_1.png)
![WandB Run](/docs/wandb_example_2.png)
![WandB Run](/docs/wandb_example_3.png)
![WandB Run](/docs/wandb_example_4.png)



