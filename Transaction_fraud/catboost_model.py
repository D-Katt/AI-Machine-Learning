"""Credit card fraud detection model. Cross-validation
of CatBoost classification models with default parameters.
Parameter optimization with buit-in Grid-Search.
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
from catboost import CatBoostClassifier

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

PATH = '../input/creditcardfraud/creditcard.csv'

SEED = 0
N_FOLDS = 3

# --------------------------- Functions -------------------------------


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

# --------------- Cross-Validating CatBoost with default parameters -------------------

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

    model_cb = CatBoostClassifier(eval_metric='AUC')
    model_cb.fit(x_train, y_train, eval_set=[(x_test, y_test)],
                 early_stopping_rounds=20, use_best_model=True,
                 verbose=0)

    scores.append(model_cb.best_score_['validation']['AUC'])
    model_cb.save_model(f'catboost{len(scores)}.cbm')

    print(f'Completed training model {len(scores)}.')

print('Average AUC score:', np.mean(scores))
for i, score in enumerate(scores):
    print(f'Model {i} AUC score: {score}')

# Display feature importance.
importance = pd.DataFrame({
    'features': x.columns,
    'importance': model_cb.feature_importances_
})
importance.sort_values(by='importance', inplace=True)

plt.figure(figsize=(12, 16))
plt.barh(importance['features'], importance['importance'])
plt.title('CatBoost Feature Importance')
plt.show()

# Parameters of the last trained model
print(model_cb.get_all_params())

# ----------------------- Hyper-Parameter Grid-Search --------------------------

model_cb = CatBoostClassifier(eval_metric='AUC')

params = {'learning_rate': [0.1, 0.2, 0.3, 0.4],
          'depth': [4, 5, 6, 7]}

grid = model_cb.grid_search(params, X=x, y=y, stratified=True, refit=True)

# This grid-search method fits the model, which was passed to it
# and returns a dictionary of grid-search.
# Check that the model is actually fitted.
print(model_cb.is_fitted())

# Save the optimized and trained model and display the parameters.
model_cb.save_model('catboost_opt.cbm')
print('CatBoost optimal parameters:', grid['params'])
