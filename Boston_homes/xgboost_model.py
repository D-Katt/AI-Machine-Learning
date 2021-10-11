"""XGBoost regression model for house pricing.
Dataset: Boston homes
"""

import os
import random
import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor

import matplotlib.pyplot as plt

VAL_SIZE = 0.2
EARLY_ROUNDS = 50
SEED = 5

# Plots display settings
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams.update({'font.size': 14})


def set_seed(seed=42):
    """Utility function to use for reproducibility.
    :param seed: Random seed
    :return: None
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(SEED)

# Original data
data = load_boston()
X = data.data  # Input features
y = data.target  # Target value (price)
parameters = data.feature_names

# Put aside a portion of the data for validation.
x_train, x_valid, y_train, y_valid = train_test_split(
    X, y, shuffle=True, random_state=SEED
)

# Train the base model
model = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor',
                         objective='reg:squarederror', booster='gbtree')

model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)],
              eval_metric='rmse', early_stopping_rounds=EARLY_ROUNDS)

# Display feature importance.
importance = pd.DataFrame({
    'features': parameters,
    'importance': model.feature_importances_
})
importance.sort_values(by='importance', inplace=True)

plt.barh(parameters, importance['importance'])
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# Evaluate validation metrics for the base model.
prediction = model.predict(x_valid)

print('Validation metrics:\n')
print('\tMAE =', metrics.mean_absolute_error(y_valid, prediction))
print('\tMSE =', metrics.mean_squared_error(y_valid, prediction))
print('\tRMSE =', np.sqrt(metrics.mean_squared_error(y_valid, prediction)))

# Validation metrics:
# 	MAE = 2.2584978854562348
# 	MSE = 8.624804820850715
# 	RMSE = 2.93680180142459
