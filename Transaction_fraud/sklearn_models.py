"""Credit card fraud detection models. Cross-validation
of sklearn classification models with default parameters.
Parameter optimization through Grid-Search.
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

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

PATH = '../input/creditcardfraud/creditcard.csv'

SEED = 0
N_FOLDS = 3

# ----------------------- Functions ------------------------------


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

data = pd.read_csv(PATH)
print(f'Data shape: {data.shape}')
print(data.head())

# Feature engineering: temporal features
# Obtain hour from "Time" column, which represents
# seconds passed from the 1st transaction.
# Data set contains transactions for 2 days.
data['hour'] = data['Time'] // 3600
data['hour'] = data['hour'].apply(lambda x: x if x < 24 else x - 24)

# -------------- Cross-Validation with default parameters ------------------

# Tree-based models do not require scaling of input features.
# Other classification models benefit from data preprocessing:
# scaling makes training faster and ensures better convergence.
# Possible scalers include StandardScaler, MinMaxScaler, PowerTransformer and some others.
# For this particular problem we will use RobustScaler where needed.
# This scaler is recommended for data, which has outliers.

# sklearn classification models
model_gausnb = make_pipeline(RobustScaler(), GaussianNB())
model_aboost = AdaBoostClassifier()
model_svc = make_pipeline(RobustScaler(), SVC())

models = [
    ('Gaussian Naive Bayes', model_gausnb),
    ('Ada boost', model_aboost),
    ('SVC', model_svc),
]

# Input features and target labels
x = data.drop('Class', axis=1)
y = data['Class']

for name, model in models:
    score = cv_score(model, x, y, metric='roc_auc')
    print(f'{name} model AUC score: {score}')

# ------------------- Grid-Search for optimal parameters ---------------------

# Example 1: Improve Naive Bayes classifier

# Create a dictionary of parameters and values to search.
# Since we created a pipeline including scaler and classifier,
# we need to use parameter name with double underscore after model name.
params = {
    'gaussiannb__var_smoothing': [0.001, 0.1, 0.5, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]
}

# We will use Stratified K-Fold algorithm to train the models
# on several subsets of data and check the AUC score of validation subsets.
kf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)

# Define Grid-Search
grid = GridSearchCV(model_gausnb, params, cv=kf, scoring='roc_auc',
                    n_jobs=-1, refit=True)

# Display the results
grid.fit(x, y)
print('Best AUC score:', grid.best_score_)
print(grid.best_params_)

# Save the best model.
joblib.dump(grid, 'gaussiannb.pkl')

# Example 2: Improve AdaBoost classifier

# In this case we use model without preprocessing step.
# Parameters in the dictionary are named the same as in model definition.
# Parameter space is limited to reduce grid-search time.
params = {
    'n_estimators': [50, 60],
    'learning_rate': [0.7, 0.8]
}

grid = GridSearchCV(model_aboost, params, cv=kf, scoring='roc_auc',
                    n_jobs=-1, refit=True)

grid.fit(data.drop('Class', axis=1), data['Class'])
print('Best AUC score:', grid.best_score_)
print(grid.best_params_)

joblib.dump(grid, 'adaboost.pkl')
