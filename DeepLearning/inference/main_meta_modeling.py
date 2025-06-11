import argparse
from pathlib import Path
from time import time

import h5py
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from scipy.stats import randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_data(h5_file_path):
    with h5py.File(h5_file_path, "r") as hf:
        first_layer_features = np.array(hf["logits"])
        second_to_last_layer_features = np.array(hf["second_to_last_layer_features"])
        ground_truths = np.array(hf["ground_truths"])
        predictions = np.array(hf["predictions"])
        correctness = np.array(hf["correctness"])
        file_names = np.array(hf["filelist"]).astype(str)
    return first_layer_features, second_to_last_layer_features, ground_truths, predictions, correctness, file_names


def get_filename_prefixes(file_names):
    return np.array([f.split('_')[0] for f in file_names])


def evaluate_model(model, X, y, groups, param_distributions, n_iter=50):
    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }
    group_kfold = GroupKFold(n_splits=5)
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, scoring='accuracy', cv=group_kfold,
                                       n_jobs=-1, verbose=1)
    random_search.fit(X, y, groups=groups)

    best_model = random_search.best_estimator_
    scores = {metric: cross_val_score(best_model, X, y, cv=group_kfold, scoring=scorer, groups=groups) for
              metric, scorer in scorers.items()}

    return best_model, scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help=r'The extractor config file (YAML). Will use the default configuration if none supplied.',
        default=Path('configurations/configuration.yaml'))
    args = parser.parse_args()
    cfg = load_config(args.config)

    checkpoint_dir = Path(cfg['MODEL_DIR']) / cfg['MODEL_NAME']
    h5_file_path = checkpoint_dir / 'model_outputs/features.h5'

    logits, second_to_last_layer_features, ground_truths, predictions, correctness, file_names = load_data(h5_file_path)

    # Get filename prefixes for grouping
    groups = get_filename_prefixes(file_names)

    # Define the models and their hyperparameter grids
    models = {
        'LogisticRegression': (Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(max_iter=10000))
        ]), {
                                   'logreg__C': uniform(0.1, 10),
                                   'logreg__penalty': ['l2'],
                                   'logreg__solver': ['lbfgs', 'saga', 'liblinear']
                               }),
        'NaiveBayes': (Pipeline([
            ('scaler', StandardScaler()),
            ('nb', GaussianNB())
        ]), {
                           'nb__var_smoothing': uniform(1e-9, 1e-7)
                       }),
        'DecisionTree': (DecisionTreeClassifier(), {
            'max_depth': randint(5, 50),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20)
        }),
        'XGBoost': (xgb.XGBClassifier(), {
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.1),
            'n_estimators': randint(100, 800),
            'min_child_weight': randint(1, 100)
        })
    }

    results = []

    feature_sets = {
        'logits': logits,
        'second_to_last_layer_features': second_to_last_layer_features
    }

    for feature_name, features in feature_sets.items():
        for name, (model, param_distributions) in models.items():
            print(f"Evaluating {name} on {feature_name}...")
            start_time = time()
            best_model, scores = evaluate_model(model, features, correctness, groups, param_distributions)
            end_time = time()
            elapsed_time = end_time - start_time
            print(f"Best hyperparameters for {name} on {feature_name}: {best_model.get_params()}")
            result = {'Model': name, 'Features': feature_name, 'Time (s)': elapsed_time}
            for metric, score in scores.items():
                result[f'{metric}_mean'] = score.mean()
                result[f'{metric}_std'] = score.std()
            results.append(result)

            # Save the best model
            joblib.dump(best_model, checkpoint_dir / f'model_outputs/best_model_{name}_{feature_name}.joblib')

    # Generate CSV report
    results_df = pd.DataFrame(results)
    results_df.to_csv(checkpoint_dir / 'model_outputs/meta_model_evaluation_report.csv', index=False)

    print("Model evaluation and hyperparameter tuning complete. Results saved to CSV.")
