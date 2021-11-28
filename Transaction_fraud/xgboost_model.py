"""Credit card fraud detection model. Cross-validation
of XGBoost classification model with default parameters.
Parameter optimization with optuna library.
Kaggle dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud
The dataset contains credit cards transactions made by European cardholders
in two days in September 2013: a total of 492 frauds out of 284,807 transactions.
The positive class (frauds) account for 0.172% of all transactions.
Features:
V1, V2, … V28 - numerical features the principal components obtained with PCA
Time - the seconds elapsed between each transaction and the first transaction in the data set
Amount - the transaction amount
Target value:
Class - takes value 1 in case of fraud and 0 otherwise
Metric:
Area Under the Precision-Recall Curve (AUPRC)
"""

import os
import random
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

PATH = '../input/creditcardfraud/creditcard.csv'

SEED = 0
N_FOLDS = 3

# ----------------------------- Functions ---------------------------------


def set_seed(seed=42):
    """Utility function to use for reproducibility.
    :param seed: Random seed
    :return: None
    """
    np.random.seed(seed)
    random.seed(seed)
    # tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def set_display():
    """Function sets display options for charts and pd.DataFrames.
    """
    # Plots display settings
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = 12, 8
    plt.rcParams.update({'font.size': 14})
    # DataFrame display settings
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.options.display.float_format = '{:.4f}'.format


def cv_score(model, x: pd.DataFrame, y: pd.Series, metric: str) -> float:
    """Function computed cross-validation score
    for classification model based on a specified metric.
    :param model: sklearn model instance
    :param x: Original DataFrame with features and target values
    :param y: Target labels
    :param metric: Scoring metric
    :return: Mean CV score
    """
    kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    score = cross_val_score(model, x, y, scoring=metric, cv=kf, n_jobs=-1)
    return score.mean()

# --------------------- Feature engineering ----------------------------

set_seed(SEED)
set_display()

data = pd.read_csv(PATH)
print(f'Data shape: {data.shape}')
print(data.head())

# Feature engineering: temporal features
# Obtain hour from "Time" column, which represents
# seconds passed from the 1st transaction.
# Data set contains transactions for 2 days.
data['hour'] = data['Time'] // 3600
data['hour'] = data['hour'].apply(lambda x: x if x < 24 else x - 24)

# --------------- Cross-Validating XGBoost with default parameters -------------------

# Input features and target labels
x = data.drop('Class', axis=1)
y = data['Class']

# We will use Stratified K-Fold algorithm to train the models
# on several subsets of data and check the AUC score of validation subsets.
kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)

scores = []

for train_index, test_index in kf.split(x, y):

    x_train, x_test = x.iloc[train_index, :], x.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model_xgb = XGBClassifier(objective='binary:logistic')

    model_xgb.fit(x_train, y_train, eval_set=[(x_test, y_test)],
                  eval_metric='auc', early_stopping_rounds=50,
                  verbose=0)

    scores.append(model_xgb.best_score)
    model_xgb.save_model(f'xgboost{len(scores)}.bin')

    print(f'Completed training model {len(scores)}.')

print('XGBoost average AUC score:', np.mean(scores))
for i, score in enumerate(scores):
    print(f'Model {i} AUC score: {score}')

# Display feature importance.
importance = pd.DataFrame({
    'features': x.columns,
    'importance': model_xgb.feature_importances_
})
importance.sort_values(by='importance', inplace=True)

plt.figure(figsize=(12, 16))
plt.barh(importance['features'], importance['importance'])
plt.title('XGBoost Feature Importance')
plt.show()

# Default parameters of the last trained model
print(model_xgb.get_params)

# ----------------------- Hyper-Parameter Tuning -----------------------------

# Optuna library requires us to define optimization function,
# which takes a "trial" object (special class used for searching parameter space),
# input features and target values. This function is called multiple times
# and returns validation score for various parameter combinations.


def objective(trial, x, y):
    """Function performs grid-search for optimal parameters.
    :param trial: optuna trial object
    :param x: Input features
    :param y: Target values
    :return: AUC score
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
        x, y, stratify=y, test_size=.2, random_state=SEED)

    # Parameter space for all tunable arguments.
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000, 100),
        'booster': 'gbtree',
        'reg_lambda': trial.suggest_int('reg_lambda', 1, 100),
        'reg_alpha': trial.suggest_int('reg_alpha', 1, 100),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0, step=0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0, step=0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 16),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.97),
        'gamma': trial.suggest_float('gamma', 0, 20)
    }

    xgb_clf = XGBClassifier(objective='binary:logistic', **params)

    # To stop unpromising trials.
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation_0-auc')

    xgb_clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                eval_metric='auc', early_stopping_rounds=50,
                callbacks=[pruning_callback], verbose=False)

    return xgb_clf.best_score


# Initialize an object to perform search for optimal parameters.
study = optuna.create_study(
    sampler=TPESampler(seed=SEED),  # Type of sampling
    direction='maximize',  # Whether the metric should be minimized or maximized.
    study_name='xgb')

# Create lambda function to call optimization function.
func = lambda trial: objective(trial, x, y)

# Here we call lambda function, which in turn calls optimization function.
study.optimize(func, n_trials=100)

print('Best XGBoost AUC score:', study.best_value)
best_params = study.best_params
print(best_params)

# Save the study.
joblib.dump(study, 'xgb_study.pkl')
