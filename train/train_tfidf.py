import pandas as pd
import numpy as np
import optuna
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

x = pd.read_csv("final_data.csv")
y = pd.read_csv("label_data_final.csv")

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)
x_train.shape, y_train.shape

model_name = "logistic_regression"

def train_model(x_train, y_train):
    models = {
        "logistic_regression" : LogisticRegression(),
        "decision_tree" : DecisionTreeClassifier(),
        "random_forest" : RandomForestClassifier(),
        "xgboost" : XGBClassifier()
    }

    model = models[model_name]
    model.fit(x_train, y_train)

    return model

def objective(trial):
    
    if model_name == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

    elif model_name == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        }

    elif model_name == "decision_tree":
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        }

    elif model_name == "logistic_regression":
        params = {
            "C": trial.suggest_float("C", 1e-3, 10, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "max_iter": 1000
        }

    model = train_model(x_train, y_train, **params)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    return acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)

with open(f"{model_name}_params.json", "w") as file:
    json.dump(study.best_params, file)

print("Best parameters:", study.best_params)
print("Best accuracy:", study.best_value)