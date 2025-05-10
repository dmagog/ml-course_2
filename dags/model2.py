import wandb
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

from config import config
from data import get_data


def train(model, x_train, y_train):
    model.fit(x_train, y_train)


def test(model, x_test, y_test):
    y_pred = model.predict(x_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x_test)
    else:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    try:
        if y_proba is not None:
            if len(set(y_test)) > 2:
                y_test_bin = label_binarize(y_test, classes=list(set(y_test)))
                auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr")
            else:
                auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = None
    except Exception as e:
        print(f"Ошибка расчёта AUC-ROC: {e}")
        auc = None

    return acc, f1, cm, auc


