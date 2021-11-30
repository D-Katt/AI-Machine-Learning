"""Apartment price model based on listings parsed from popular web sites.
Data source: https://www.kaggle.com/mrdaniilak/russia-real-estate-20182021/
Data set contains listings for Russian Housing from 2018 till 2021
across different regions, a total of 5,477,006 samples with 12 features
and a target value ("price").
Features:
'date' - date the listing was published
'time' - exact time the listing was published
'geo_lat' - geographical coordinate of the property
'geo_lon' - geographical coordinate of the property
'region' - numerically encoded geographical area
'building_type' - numerically encoded type of the building where the apartment is located
'level' - floor the apartment is located on
'levels' - total number of storeys in the building
'rooms' - number of rooms in the apartment (-1 stands for studios with open-space layout)
'area' - total floor area of the apartment in sq. meters
'kitchen_area' - kitchen area in sq. meters
'object_type' - apartment type, where 1 stands for secondary real estate market, 11 - new building
"""

import gc
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')

PATH = '../input/russia-real-estate-20182021/all_v2.csv'

REGION_ID = 2661  # City of Saint Petersburg

MIN_AREA = 20  # Outlier range for floor area
MAX_AREA = 200

MIN_KITCHEN = 6  # Outlier range for kitchen area
MAX_KITCHEN = 30

MIN_PRICE = 1_500_000  # Outlier range for price
MAX_PRICE = 50_000_000

SEED = 15
N_FOLDS = 5

# -------------------------- Functions -------------------------------


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


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Function removes excess columns and enforces
    correct data types.
    :param df: Original DataFrame
    :return: Updated DataFrame
    """
    df.drop('time', axis=1, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    # Column actually contains -1 and -2 values presumably for studio apartments.
    df['rooms'] = df['rooms'].apply(lambda x: 0 if x < 0 else x)
    df['price'] = df['price'].abs()  # Fix negative values
    # Drop price and area outliers.
    df = df[(df['area'] <= MAX_AREA) & (df['area'] >= MIN_AREA)]
    df = df[(df['price'] <= MAX_PRICE) & (df['price'] >= MIN_PRICE)]
    # Fix kitchen area outliers.
    # At first, replace all outliers with 0.
    df.loc[(df['kitchen_area'] >= MAX_KITCHEN) | (df['area'] <= MIN_AREA), 'kitchen_area'] = 0
    # Then calculate kitchen area based on the floor area, except for studios.
    erea_mean, kitchen_mean = df[['area', 'kitchen_area']].quantile(0.5)
    kitchen_share = kitchen_mean / erea_mean
    df.loc[(df['kitchen_area'] == 0) & (df['rooms'] != 0), 'kitchen_area'] = \
        df.loc[(df['kitchen_area'] == 0) & (df['rooms'] != 0), 'area'] * kitchen_share

    return df


def select_region(df: pd.DataFrame) -> pd.DataFrame:
    """Function selects the listings belonging to a specified region.
    :param df: Original DataFrame with all listings
    :return: Filtered DataFrame
    """
    df = df[df['region'] == REGION_ID]
    df.drop('region', axis=1, inplace=True)
    print(f'Selected {len(df)} samples in region {REGION_ID}.')
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Replace "date" with numeric features for year and month.
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df.drop('date', axis=1, inplace=True)
    # Apartment floor in relation to total number of floors.
    df['level_to_levels'] = df['level'] / df['levels']
    # Average size of room in the apartment.
    df['area_to_rooms'] = (df['area'] / df['rooms']).abs()
    # Fix division by zero.
    df.loc[df['area_to_rooms'] == np.inf, 'area_to_rooms'] = \
        df.loc[df['area_to_rooms'] == np.inf, 'area']
    return df


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


# ----------------------------- EDA ----------------------------------

set_display()

data = pd.read_csv(PATH)

print(f'Data shape: {data.shape}')
print(data.head())

print(data.info())
print(data.describe())

# Fix errors and remove outliers.
data = data.pipe(clean_data)
print(data.head())

building_types = data['building_type'].value_counts()
plt.pie(building_types.values, labels=building_types.index, autopct='%1.1f%%')
plt.title('Building Types')
plt.show()

apartment_types = data['object_type'].value_counts()
plt.pie(apartment_types.values, labels=apartment_types.index, autopct='%1.1f%%')
plt.title('Apartment Types')
plt.show()

rooms = data['rooms'].value_counts()
plt.pie(rooms.values, labels=rooms.index, autopct='%1.1f%%')
plt.title('Number of Rooms')
plt.show()

pos = 0
for pos, feature in enumerate(['area', 'kitchen_area']):
    sp = plt.subplot(1, 2, pos + 1)
    plt.hist(data[feature], bins=20)
    plt.title(f'{feature} Distribution')
plt.show()

pos = 0
for pos, feature in enumerate(['level', 'levels']):
    levels = data[feature].value_counts()
    sp = plt.subplot(1, 2, pos + 1)
    plt.bar(levels.index, levels.values)
    plt.title(f'{feature}')
plt.show()

# Regions are encoded with numeric IDs.
regions = data['region'].value_counts()
print(regions.index)

plt.hist(regions.values, bins=10)
plt.title('Listings by Region')
plt.show()

# Find out what regions are represented in the data set.
for region in data['region'].unique():
    subset = data[data['region'] == region]
    lat, lon = np.round(subset[['geo_lat', 'geo_lon']].mean(), 2)
    print(f'Region {region}: latitude = {lat}, longitude = {lon}')

avg_prices = data.groupby(by='region')['price'].mean()
plt.hist(avg_prices.values, bins=10)
plt.xlabel('Rubles')
plt.ylabel('Frequency')
plt.title('Average Price by Region')
plt.show()

# ------------------- Feature Engineering --------------------------

data = data.pipe(select_region).pipe(add_features)
print(data.head())

gc.collect()

correlation = data.corr()
ax = sns.heatmap(correlation, center=0, cmap='RdBu_r')
l, r = ax.get_ylim()
ax.set_ylim(l + 0.5, r - 0.5)
plt.yticks(rotation=0)
plt.title('Correlation Matrix')
plt.show()

y = data.pop('price')

# ------------------- Models Cross-Validation --------------------------

set_seed(SEED)

kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)

# XGBoost model

scores = []

for train_index, test_index in kf.split(data, y):

    x_train, x_test = data.iloc[train_index, :], data.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model_xgb = XGBRegressor(objective='reg:squarederror')

    model_xgb.fit(x_train, y_train, eval_set=[(x_test, y_test)],
                  eval_metric='rmse', early_stopping_rounds=50,
                  verbose=0)

    scores.append(model_xgb.best_score)
    model_xgb.save_model(f'xgboost{len(scores)}.bin')

    print(f'Completed training model {len(scores)}.')

print('XGBoost average RMSE:', np.mean(scores))
for i, score in enumerate(scores):
    print(f'Model {i} RMSE: {score}')

# Display feature importance.
importance = pd.DataFrame({
    'features': data.columns,
    'importance': model_xgb.feature_importances_
})
importance.sort_values(by='importance', inplace=True)

plt.figure(figsize=(12, 16))
plt.barh(importance['features'], importance['importance'])
plt.title('XGBoost Feature Importance')
plt.show()

# Default parameters of the last trained model
print(model_xgb.get_params)

# LightGBM model

scores = []

for train_index, test_index in kf.split(data, y):

    x_train, x_test = data.iloc[train_index, :], data.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model_lgb = LGBMRegressor(objective='regression', metrics='rmse')

    model_lgb.fit(x_train, y_train, eval_set=(x_test, y_test),
                  eval_metric='rmse', early_stopping_rounds=50,
                  categorical_feature=['building_type', 'object_type', 'month'],
                  verbose=0)

    scores.append(model_lgb.best_score_['valid_0']['rmse'])
    model_lgb.booster_.save_model(f'lgbm{len(scores)}.txt',
                                  num_iteration=model_lgb.best_iteration_)

    print(f'Completed training model {len(scores)}.')

print('LGBM average RMSE:', np.mean(scores))
for i, score in enumerate(scores):
    print(f'Model {i} RMSE: {score}')

importance = pd.DataFrame({
    'features': data.columns,
    'importance': model_lgb.feature_importances_
})
importance.sort_values(by='importance', inplace=True)

plt.figure(figsize=(12, 16))
plt.barh(importance['features'], importance['importance'])
plt.title('LGBM Feature Importance')
plt.show()

# Default parameters of the last trained model
print(model_lgb.get_params())

# CatBoost model

scores = []

for train_index, test_index in kf.split(data, y):

    x_train, x_test = data.iloc[train_index, :], data.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model_cb = CatBoostRegressor(eval_metric='RMSE',
                                 cat_features=['building_type', 'object_type', 'month'])
    model_cb.fit(x_train, y_train, eval_set=[(x_test, y_test)],
                 early_stopping_rounds=20, use_best_model=True,
                 verbose=0)

    scores.append(model_cb.best_score_['validation']['RMSE'])
    model_cb.save_model(f'catboost{len(scores)}.cbm')

    print(f'Completed training model {len(scores)}.')

print('Average RMSE:', np.mean(scores))
for i, score in enumerate(scores):
    print(f'Model {i} RMSE: {score}')

# Display feature importance.
importance = pd.DataFrame({
    'features': data.columns,
    'importance': model_cb.feature_importances_
})
importance.sort_values(by='importance', inplace=True)

plt.figure(figsize=(12, 16))
plt.barh(importance['features'], importance['importance'])
plt.title('CatBoost Feature Importance')
plt.show()

# Parameters of the last trained model
print(model_cb.get_all_params())
