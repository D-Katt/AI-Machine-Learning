"""Linear regression house pricing model.
Dataset: Boston homes
Model: sklearn LinearRegression
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

VAL_SIZE = 0.2
SEED = 5

# Plots display settings
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams.update({'font.size': 14})

# Original data
data = load_boston()
X = data.data  # Input features
y = data.target  # Target value (price)
parameters = data.feature_names

# Distribution of the target values.
plt.hist(y, bins=20)
plt.axvline(np.mean(y), label='Mean', color='red')
plt.axvline(np.median(y), label='Median', color='green')
plt.legend()
plt.ylabel('Frequency')
plt.xlabel('Price, 000 USD')
plt.title('Target Values')
plt.tight_layout()
plt.savefig('targets.png')
plt.show()

# Correlation between the features.
data = pd.DataFrame(X, columns=parameters)
data['price'] = y
ax = sns.heatmap(data.corr(), annot=True, cmap=plt.cm.Reds, fmt='0.3f')
l, r = ax.get_ylim()
ax.set_ylim(l + 0.5, r - 0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation.png')
plt.show()

# Put aside a portion of the data for testing.
x_train, x_test, y_train, y_test = train_test_split(
    X, y, shuffle=True, random_state=SEED
)

# Pipeline with a scaler and linear regression model.
pipe = make_pipeline(StandardScaler(), LinearRegression())
pipe.fit(x_train, y_train)

# Save the pipeline to a file.
joblib.dump(pipe, 'pipe.joblib')

# Display model parameters
print('Intercept =', pipe[1].intercept_)
print('Linear coefficients:')
for par, coef in zip(parameters, pipe[1].coef_):
    print(f'\t{par}: {coef}')

# Mean Squared Error in 10-fold cross validation.
cv_mse = -cross_val_score(pipe, x_train, y_train, cv=10,
                          scoring='neg_mean_squared_error').mean()
print(f'Cross-validation:\n'
      f'\tMSE = {cv_mse}\n'
      f'\tRMSE = {np.sqrt(cv_mse)}')

# Cross-validation:
# 	MSE = 24.09109976980676
# 	RMSE = 4.908268510361547

# Forecast for the test set.
prediction = pipe.predict(x_test)
print('Test:\n')
print('\tMAE =', metrics.mean_absolute_error(y_test, prediction))
print('\tMSE =', metrics.mean_squared_error(y_test, prediction))
print('\tRMSE =', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

# Test:
# 	MAE = 3.2900183526885503
# 	MSE = 24.274608311687857
# 	RMSE = 4.926926862831217

# Actual vs. predicted house price.
x_axis = range(len(prediction))
plt.plot(x_axis, prediction, label='Forecast')
plt.plot(x_axis, y_test, label='Actual data')
plt.ylabel('House price, 000 UDS')
plt.xlabel('Test samples')
plt.legend()
plt.title('Model Predictions')
plt.tight_layout()
plt.savefig('lr_prediction.png')
plt.show()

# Errors as a percentage of actual prices.
errors = (prediction - y_test) / y_test * 100

plt.hist(errors, bins=20)
plt.ylabel('Frequency')
plt.xlabel('Error, %')
plt.title('Model Errors')
plt.tight_layout()
plt.savefig('lr_errors.png')
plt.show()
