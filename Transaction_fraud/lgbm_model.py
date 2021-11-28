"""Credit card fraud detection model. Cross-validation
of LGBM classification models with default parameters.
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

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

PATH = '../input/creditcardfraud/creditcard.csv'

SEED = 0
N_FOLDS = 3

# ------------------------- Functions ---------------------------


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

# --------------- Cross-Validating LGBM with default parameters -------------------

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

    model_lgb = LGBMClassifier(objective='binary', metrics='auc')

    model_lgb.fit(x_train, y_train, eval_set=(x_test, y_test),
                  eval_metric='auc', early_stopping_rounds=50,
                  verbose=0)

    scores.append(model_lgb.best_score_['valid_0']['auc'])
    model_lgb.booster_.save_model(f'lgbm{len(scores)}.txt',
                                  num_iteration=model_lgb.best_iteration_)

    print(f'Completed training model {len(scores)}.')

print('LGBM average AUC score:', np.mean(scores))
for i, score in enumerate(scores):
    print(f'Model {i} AUC score: {score}')

importance = pd.DataFrame({
    'features': x.columns,
    'importance': model_lgb.feature_importances_
})
importance.sort_values(by='importance', inplace=True)

plt.figure(figsize=(12, 16))
plt.barh(importance['features'], importance['importance'])
plt.title('LGBM Feature Importance')
plt.show()

# Default parameters of the last trained model
print(model_lgb.get_params())
