import pandas as pd
import numpy as np
import optuna
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from loguru import logger

logger.info("Starting the training script")
x = pd.read_csv("final_data.csv")
y = pd.read_csv("label_data_final.csv")
logger.success("Data loaded successfully")
logger.info(f"Feature data shape: {x.shape}")
logger.info(f"Label data shape: {y.shape}")
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)
logger.success("Data split into training and testing sets")
logger.info(f"Training feature shape: {x_train.shape}")
logger.info(f"Testing feature shape: {x_test.shape}")
logger.info(f"Training label shape: {y_train.shape}")
logger.info(f"Testing label shape: {y_test.shape}")

model_name = "logistic_regression"
logger.info(f"Selected model: {model_name}")


def train_model(x_train, y_train):
    logger.info(f"Training model: {model_name}")
    models = {
        "logistic_regression": LogisticRegression(),
        "decision_tree": DecisionTreeClassifier(),
        "random_forest": RandomForestClassifier(),
        "xgboost": XGBClassifier()
    }

    model = models[model_name]
    model.fit(x_train, y_train)

    return model


def objective(trial):
    logger.info(f"Trial number: {trial.number}")
    if model_name == "xgboost":
        logger.info("Optimizing XGBoost parameters")
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

    elif model_name == "random_forest":
        logger.info("Optimizing Random Forest parameters")
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        }

    elif model_name == "decision_tree":
        logger.info("Optimizing Decision Tree parameters")
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        }

    elif model_name == "logistic_regression":
        logger.info("Optimizing Logistic Regression parameters")
        params = {
            "C": trial.suggest_float("C", 1e-3, 10, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "max_iter": 1000
        }

    model = train_model(x_train, y_train, **params)
    logger.success("Model training completed")
    y_pred = model.predict(x_test)
    logger.info("Model evaluation completed")
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {acc}")

    return acc


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)

with open(f"{model_name}_params.json", "w") as file:
    json.dump(study.best_params, file)

logger.success("Best parameters:", study.best_params)
logger.success("Best accuracy:", study.best_value)
