from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import optuna
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from loguru import logger

MODEL_BUILDERS = {
    'logistic_regression': lambda **p: LogisticRegression(**p),
    'decision_tree': lambda **p: DecisionTreeClassifier(**p),
    'random_forest': lambda **p: RandomForestClassifier(**p),
    'xgboost': lambda **p: XGBClassifier(use_label_encoder=False, eval_metric='logloss', **p),
}

PARAM_SPACES = {
    'logistic_regression': {
        'C': (1e-3, 10, 'log'),
        'solver': ['lbfgs', 'saga'],
        'max_iter': 1000,
    },
    'decision_tree': {
        'max_depth': (2, 20),
        'min_samples_split': (2, 10),
    },
    'random_forest': {
        'n_estimators': (50, 300),
        'max_depth': (2, 20),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 4),
    },
    'xgboost': {
        'n_estimators': (50, 300),
        'max_depth': (2, 12),
        'learning_rate': (0.01, 0.3, 'log'),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
    }
}


def build_params(trial, model_name):
    params = {}
    space = PARAM_SPACES[model_name]
    for k, v in space.items():
        if isinstance(v, tuple) and len(v) == 3 and v[2] == 'log':
            params[k] = trial.suggest_float(k, v[0], v[1], log=True)
        elif isinstance(v, tuple) and len(v) == 2:
            if all(isinstance(x, int) for x in v):
                params[k] = trial.suggest_int(k, v[0], v[1])
            else:
                params[k] = trial.suggest_float(k, v[0], v[1])
        elif isinstance(v, list):
            params[k] = trial.suggest_categorical(k, v)
        else:
            params[k] = v
    return params


class RunExperimentAPIView(APIView):
    """POST endpoint to run an experiment."""

    def get(self, request):
        logger.info("Received GET request for experiment")
        model_name = request.query_params.get('model_name', 'logistic_regression')
        n_trials = int(request.query_params.get('n_trials', 10))
        optimize = request.query_params.get('optimize', True)

        if model_name not in MODEL_BUILDERS:
            return Response({'error': 'Unsupported model_name'}, status=status.HTTP_400_BAD_REQUEST)

        x = sp.sparse.load_npz('/tmp/sparse_matrix.npz')
        y = sp.sparse.load_npz('/tmp/labels.npz').toarray().ravel()
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)

        logger.info(f"Running stateless experiment with model={model_name} optimize={optimize}")

        best_params = {}
        best_score = None

        if optimize:
            def objective(trial):
                params = build_params(trial, model_name)
                model = MODEL_BUILDERS[model_name](**params)
                model.fit(x_train, y_train)
                preds = model.predict(x_test)
                return accuracy_score(y_test, preds)

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            best_score = study.best_value
        else:
            # Use defaults
            if model_name == 'logistic_regression':
                best_params = {'max_iter': 1000}
            else:
                best_params = {}

        model = MODEL_BUILDERS[model_name](**best_params)
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        result = {
            'model_name': model_name,
            'optimized': bool(optimize),
            'requested_trials': n_trials if optimize else 0,
            'best_params': best_params,
            'search_best_score': best_score if optimize else acc,
            'final_accuracy': acc,
            'classification_report': report,
        }
        return Response(result)
