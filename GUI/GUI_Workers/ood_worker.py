from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from scipy.stats import randint
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import uniform
from sklearn.tree import DecisionTreeClassifier


class OODWorker(QThread):
    finished = pyqtSignal(object, object, object)  # Signal to emit results

    def __init__(self, features, correctness_mask):
        super().__init__()
        self.features = features
        self.correctness_mask = correctness_mask
        self.models_and_params = [
            {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': randint(1, 20),
                    'weights': ['uniform', 'distance']
                }
            },
            {
                'model': RandomForestClassifier(),
                'params': {
                    'n_estimators': randint(10, 100),
                    'max_depth': [None, 10, 20, 30, 40],
                    'min_samples_split': randint(2, 11),
                    'min_samples_leaf': randint(1, 11)
                }
            },
            {
                'model': LogisticRegression(max_iter=1000),
                'params': {
                    'C': uniform(0.1, 10),
                    'penalty': ['l2']
                }
            },
            {
                'model': DecisionTreeClassifier(),
                'params': {
                    'max_depth': [None, 10, 20, 30, 40],
                    'min_samples_split': randint(2, 11),
                    'min_samples_leaf': randint(1, 11)
                }
            },
            {
                'model': GradientBoostingClassifier(),
                'params': {
                    'n_estimators': randint(50, 100),
                    'learning_rate': uniform(0.01, 0.3),
                    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10]
                }
            },
        ]

    def run(self):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        best_overall_score = -np.inf
        best_overall_model = None
        best_overall_params = None

        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score)
        }

        for model_entry in self.models_and_params:
            model = model_entry['model']
            param_dist = model_entry['params']

            print(f"Testing model: {model.__class__.__name__}")

            # Use the defined scoring metrics in RandomizedSearchCV
            random_search = RandomizedSearchCV(
                model,
                param_distributions=param_dist,
                n_iter=10,
                cv=cv,
                scoring=scoring,
                refit='f1',
                verbose=0,
                n_jobs=1,
                random_state=42
            )
            random_search.fit(self.features, self.correctness_mask)

            print("Best Scores for Each Metric for", model.__class__.__name__, ":")
            for scorer in scoring:
                print(f"{scorer}: {random_search.cv_results_['mean_test_' + scorer][random_search.best_index_]}")

            # Check if this model has the highest F1 score seen so far
            if random_search.best_score_ > best_overall_score:
                best_overall_score = random_search.best_score_
                best_overall_params = random_search.best_params_
                best_overall_model = random_search.best_estimator_

        # Emit the results without the classification report
        self.finished.emit(best_overall_params, best_overall_score, best_overall_model)
