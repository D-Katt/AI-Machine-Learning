"""Analysis and modeling of demand for bikes at a bike sharing service in a city of Chicago.
Data set contains description of bike rides starting from April 2020 till April 2021 -
a total of 3,826,978 samples and 713 bike stations.
Each sample contains start and end station name, ID, latitude, longitude, time,
bike type and user category (member of casual). User ID is not available.
Two tasks are being solved with this data set:
- Predict total number of bikes rented at each bike station daily.
  Input features include location, temporal features, categorical features
  defining station type and lagged target values (demand in previous periods).
- Predict ride duration and end point coordinates each individual ride.
  Input features include location, temporal features, categorical features
  defining bike type, user type and station type.
Bike sharing data set: https://www.kaggle.com/kevinvenza/cyclistic-bike-share
"""

import os
import gc
import random
import json
import glob

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

# Directory with multiple data files.
DIRECTORY = '../input/cyclistic-bike-share'

# -------------------------------- Functions -------------------------------------


def set_seed(seed=42):
    """Utility function to use for reproducibility.
    :param seed: Random seed
    :return: None
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
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


def get_category_encoding_layer(cat_feature: np.array, dtype: str,
                                max_tokens=None):
    """Function creates category encoding layer
    with string or integer lookup index.
    :param cat_feature: Array containing categorical input data for one feature
    :param dtype: String describing data type of the categorical feature (one of 'string' or 'int64')
    :param max_tokens: Maximum number of tokens in the lookup index
    :return: Lambda function with categorical encoding layers and lookup index
    """
    # Lookup layer which turns strings or integers into indices
    if dtype == 'string':
        index = tf.keras.layers.experimental.preprocessing.StringLookup(
            max_tokens=max_tokens)
    else:  # 'int64'
        index = tf.keras.layers.experimental.preprocessing.IntegerLookup(
            max_tokens=max_tokens)
    # Learn the data scale
    index.adapt(cat_feature)
    # Category encoding layer
    encoder = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
        max_tokens=index.vocab_size())
    return lambda feature: encoder(index(feature))


set_seed()
set_display()

# ----------------------------- Load the Data -----------------------------

# Read the data from all csv files.
paths = glob.glob(f'{DIRECTORY}/*.csv')
data = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
print(f'Data shape: {data.shape}')

# ID and date columns have object data type.
print(data.dtypes)

# There are missing values in station name and ID columns
# and some latitude and longitude columns.
print(data.isna().sum())

# --------------------- Data Processing and Analysis --------------------------

# Cleaning data types for date columns.
data['started_at'] = pd.to_datetime(data['started_at'])
data['ended_at'] = pd.to_datetime(data['ended_at'])

# Start and end coordinates vary in a narrow range -
# all the bike stations are located in the city of Chicago.
stats = data.describe()
print(stats)

# Data covers 13 months starting from April 2021 till April 2021.
print(f'Time span: {data["started_at"].min()} - {data["started_at"].max()}')

# Feature analysis
for feature in ('start_station_name', 'end_station_name'):
    print(f'Feature "{feature}": {data[feature].nunique()} unique values')

# Categorical features
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
data['rideable_type'].value_counts().plot.pie(
    autopct='%1.2f%%', ax=ax[0], fontsize=12, startangle=135)
data['member_casual'].value_counts().plot.pie(
    autopct='%1.2f%%', ax=ax[1], fontsize=12, startangle=135)
plt.suptitle('Categorical Features', fontsize=20)
plt.show()

# Frequency of start and end points in the data set.
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].hist(data['start_station_name'].value_counts(), bins=20)
ax[1].hist(data['end_station_name'].value_counts(), bins=20)
ax[0].set(xlabel='Start station frequency')
ax[1].set(xlabel='End station frequency')
plt.suptitle('Bike Station Popularity', fontsize=20)
plt.show()

# Range of coordinates
data.hist(bins=20, figsize=(16, 16))
plt.suptitle('Seasonality and Station Coordinates')
plt.show()

# --------------------------- Feature engineering ---------------------------------

# Rent duration
data['duration'] = (data['ended_at'] - data['started_at']) / np.timedelta64(60, 's')  # In minutes

# Location features
data['same_station'] = (data['start_station_name'] == data['end_station_name']).astype(int)
data['park'] = data['start_station_name'].str.lower().str.contains('park')
data['park'] = data['park'].fillna(0).astype(int)

# Temporal features
data['year'] = data['started_at'].dt.year
data['month'] = data['started_at'].dt.month
data['weekofyear'] = data['started_at'].dt.isocalendar().week
data['dayofyear'] = data['started_at'].dt.dayofyear
data['day'] = data['started_at'].dt.day
data['dayofweek'] = data['started_at'].dt.dayofweek
data['hour'] = data['started_at'].dt.hour

# Demand analysis with respect to rent duration and start / end points.
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
data['same_station'].value_counts().plot.pie(
    explode=[0, 0.25], autopct='%1.2f%%', ax=ax[0],
    fontsize=12, startangle=135)
data['duration'].hist(bins=20, ax=ax[1], log=True)
plt.suptitle('Bike Demand', fontsize=20)
ax[1].set(ylabel='Frequency')
ax[1].set(xlabel='Duration, minutes')
ax[1].legend()
plt.show()

data['park'].value_counts().plot.pie(
    autopct='%1.2f%%', ax=ax[1], fontsize=12, startangle=135)
plt.suptitle('Proximity to Park', fontsize=20)
plt.show()

# Since the data covers 13 months, we drop the latest month
# to be able to show how demand is distributed throughout the year.
for feature in ('month', 'weekofyear', 'dayofyear', 'day',
                'dayofweek', 'hour'):
    grouped_data = data.groupby(by=feature)['ride_id'].count()
    grouped_data = grouped_data / grouped_data.sum()
    plt.bar(grouped_data.index, grouped_data.values)
    plt.title(f'Bike Demand by {feature}')
    plt.ylabel('Percentage of total demand')
    plt.show()

# --------------------- Convert data into daily format ------------------------

# Sum up the number of rents per day for each bike station.
daily_data = data.groupby(by=['start_station_name', 'dayofyear']).agg(
    {'ride_id': 'count', 'start_lat': 'min', 'start_lng': 'min',
     'park': 'min', 'year': 'min', 'month': 'min',
     'weekofyear': 'min', 'day': 'min', 'dayofweek': 'min'}).reset_index()

daily_data = daily_data.rename(columns={'ride_id': 'n_bikes'})

print('Data shape:', daily_data.shape)
print(daily_data.describe())

daily_data['n_bikes'].hist(bins=100)
plt.ylabel('Frequency')
plt.xlabel('N bikes rented')
plt.title('Daily Bike Rents')
plt.show()

# Days when no bikes were rented at the station are missing in the data set.
# Create a new DataFrame for all possible combinations of station name
# and days of the period from April 2020 till April 2021.
stations = daily_data['start_station_name'].unique().tolist()
print('Unique stations:', len(stations))

all_days_stations = pd.DataFrame(columns=stations)
all_days_stations['date'] = pd.date_range(start='2020-04-01', end='2021-04-30', freq='D')
all_days_stations = all_days_stations.fillna(0)

print('Data shape before melt:', all_days_stations.shape)
print('Dates range:', all_days_stations['date'].min(), all_days_stations['date'].max())

# We need station name, year and day of year columns
# to join this data with the original DataFrame.
all_days_stations['year'] = all_days_stations['date'].dt.year
all_days_stations['dayofyear'] = all_days_stations['date'].dt.dayofyear

all_days_stations = pd.melt(
    all_days_stations, id_vars=['year', 'dayofyear'],
    value_vars=stations, var_name='start_station_name', value_name='n_bikes')

print('Data shape after melt:', all_days_stations.shape)

# Fill in missing values.
daily_data['n_bikes'] = daily_data['n_bikes'].fillna(0)

# Fill in missing values for location and time features
# by copying previous values in a sorted DataFrame.
daily_data.sort_values(
    by=['start_station_name', 'n_bikes'],
    ascending=False, inplace=True)
daily_data[['start_lat', 'start_lng', 'park']] = daily_data[
    ['start_lat', 'start_lng', 'park']].fillna(method='ffill')

daily_data.sort_values(
    by=['year', 'dayofyear', 'n_bikes'],
    ascending=False, inplace=True)
daily_data[['year', 'month', 'weekofyear', 'day', 'dayofweek']] = daily_data[
    ['year', 'month', 'weekofyear', 'day', 'dayofweek']].fillna(method='ffill')

# Fix data types
for feature in ('weekofyear', 'month', 'year'):
    daily_data[feature] = daily_data[feature].astype(int)

# Demand Heat-Map
demand_by_period = daily_data.pivot_table(
    index='year', columns='month', values='n_bikes', aggfunc='sum')
ax = sns.heatmap(demand_by_period, center=0, annot=False, cmap='RdBu_r')
l, r = ax.get_ylim()
ax.set_ylim(l + 0.5, r - 0.5)
plt.yticks(rotation=0)
plt.title('Demand by Period')
plt.show()

demand_by_station = daily_data.pivot_table(
    index='start_station_name', columns='month', values='n_bikes', aggfunc='sum')
ax = sns.heatmap(demand_by_station, center=0, annot=False, cmap='RdBu_r')
l, r = ax.get_ylim()
ax.set_ylim(l + 0.5, r - 0.5)
plt.yticks(rotation=0)
plt.title('Demand by Station')
plt.show()

# Feature engineering for daily data

# Group the bike stations into clusters according to their popularity.
# To avoid data leakage we do not use the entire data set to group the stations.
# We take the data up till March 2021 and bin total number of rent events
# for each station during that period.
rents_by_station = daily_data[
    ~((daily_data['year'] == 2021) & (daily_data['month'] == 4))
].groupby(by='start_station_name')['n_bikes'].sum()

bins = pd.cut(rents_by_station, bins=10, labels=[i for i in range(10)])
cluster_dict = dict(zip(rents_by_station.index, bins))

with open('station_clusters.json', 'w') as f:
    json.dump(cluster_dict, f, indent=2)

daily_data['cluster'] = daily_data['start_station_name'].apply(lambda x: cluster_dict[x])
daily_data.head()

# Distribution of stations between clusters.
clusters = daily_data['cluster'].value_counts() / 395  # 395 days for each bike station.
plt.bar(clusters.index, clusters.values)
plt.ylabel('Number of stations')
plt.xlabel('Cluster IDs')
plt.title('Station Clusters')
plt.show()

# Lagged target values: the number of bikes rented at the same station
# the previous day and 7 days before the predicted day.
# To get the lagged target values for each station separately
# we sort the data by station and year and day of year
# and shift target values in a grouped DataFrame.
daily_data.sort_values(
    by=['start_station_name', 'year', 'dayofyear'],
    inplace=True)

for num in (1, 7):
    daily_data[f'lag_{num}'] = daily_data.groupby(
        by='start_station_name')['n_bikes'].shift(num)

# Rolling mean of the target for the last 7 days before the predicted day.
daily_data['rol_week'] = daily_data.groupby(
        by='start_station_name')['lag_1'].transform(lambda x: x.rolling(7).mean())

# Save the data for future use.
daily_data.to_csv('daily_data.csv', index=False)

# Correlation matrix
correlation = daily_data.corr()
ax = sns.heatmap(correlation, center=0, annot=True, cmap='RdBu_r', fmt='0.2f')
l, r = ax.get_ylim()
ax.set_ylim(l + 0.5, r - 0.5)
plt.yticks(rotation=0)
plt.title('Correlation Matrix')
plt.show()

# Select row indexes for every fourth week of the data set for validation.
valid_idx = daily_data[daily_data['weekofyear'] % 4 == 0].index
data_valid = daily_data.loc[valid_idx, :]
data_train = daily_data.drop(data_valid.index)
print(f'Train data shape: {data_train.shape}\n'
      f'Validation data shape: {data_valid.shape}')

y_train = data_train.pop('n_bikes')
y_valid = data_valid.pop('n_bikes')

#########################################################################################
#                Predict daily demand for bikes per station
#########################################################################################

# XGBoost model

# For this model we drop station names assuming that geographical coordinates,
# station clusters, temporal features and lagging target values will be enough
# to predict daily bike rents.
# Testing showed that encoding 713 bike station names and using this as an input feature
# does not improve XGBoost model accuracy but decreases the impact of other useful features.

data_train.drop('start_station_name', axis=1, inplace=True)
data_valid.drop('start_station_name', axis=1, inplace=True)

print(data_train.head())

EARLY_ROUNDS = 50

model = XGBRegressor(
    n_estimators=500,
    tree_method='gpu_hist', predictor='gpu_predictor',  # With CPU the model is more accurate.
    objective='reg:squarederror', booster='gbtree')

model.fit(data_train, y_train, eval_set=[(data_valid, y_valid)],
          eval_metric='rmse', early_stopping_rounds=EARLY_ROUNDS)

# Save the regressor.
model.save_model('xgb_demand_model.bin')

print(f'Validation RMSE = {model.best_score}')

# Check the feature importance.
importance = pd.DataFrame({
    'features': data_train.columns,
    'importance': model.feature_importances_
})
importance.sort_values(by='importance', inplace=True)

plt.barh(importance['features'], importance['importance'])
plt.title('XGBoost Feature Importance')
plt.show()

# Check the actual magnitude of errors and their distribution.
data_valid['prediction'] = model.predict(data_valid)
data_valid['n_bikes'] = y_valid
data_valid['mae'] = (data_valid['prediction'] - data_valid['n_bikes']).abs()

print('XGBoost Validation MAE:', data_valid['mae'].mean())

plt.hist(data_valid['mae'], bins=100)
plt.xlabel('Error, bikes a day')
plt.ylabel('Frequency')
plt.title('XGBoost Mean Absolute Error')
plt.show()

# Neural network with TensorFlow

# Drop samples with NaN values from the daily data.
daily_data = daily_data.dropna()

# Update training and validation subsets.
valid_idx = daily_data[daily_data['weekofyear'] % 4 == 0].index
data_valid = daily_data.loc[valid_idx, :]
data_train = daily_data.drop(data_valid.index)

print(f'Train data shape: {data_train.shape}\n'
      f'Validation data shape: {data_valid.shape}')

print(data_train.head())

# Move categorical columns into a separate input array.
data_train_cat = data_train.pop('start_station_name')
data_valid_cat = data_valid.pop('start_station_name')

# Move target values into separate variables.
y_train = data_train.pop('n_bikes')
y_valid = data_valid.pop('n_bikes')

# Create normalization layer and adapt it to the numerical data.
normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=None)
normalizer.adapt(data_train.sample(n=30_000))

# Create  encoding layer for categorical feature.
encoding_layer = get_category_encoding_layer(
    data_train_cat.drop_duplicates().values, dtype='string')

# Model input layers
num_input = tf.keras.Input(shape=(data_train.shape[1],), dtype=tf.float32)
cat_input = tf.keras.Input(shape=(1,), dtype='string')

# Process inputs separately and combine.
x_1 = normalizer(num_input)
x_2 = encoding_layer(cat_input)
x = tf.keras.layers.concatenate([x_1, x_2])

x = tf.keras.layers.Dense(
    128, activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.Dense(
    64, activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.Dense(
    32, activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

output = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model([num_input, cat_input], output)

model.compile(optimizer='adam',
              loss='mse',
              metrics=[tf.keras.metrics.MeanAbsoluteError(),
                       tf.keras.metrics.RootMeanSquaredError()]
              )

# Visualize the model graph
tf.keras.utils.plot_model(model, show_shapes=True, show_dtype=True)

history = model.fit(x=[data_train, data_train_cat], y=y_train,
                    epochs=20, batch_size=64)  # Small batch size regularizes the training.

# Save the model
model.save('tf_demand_model')

# Check the magnitude of validation errors and their distribution.
data_valid['prediction'] = model.predict(x=[data_valid, data_valid_cat], batch_size=512)
data_valid['n_bikes'] = y_valid

data_valid['mae'] = (data_valid['prediction'] - data_valid['n_bikes']).abs()
print('Validation MAE =', data_valid['mae'].mean())

plt.hist(data_valid['mae'], bins=100)
plt.xlabel('Error, bikes a day')
plt.ylabel('Frequency')
plt.title('Neural Net Mean Absolute Error')
plt.show()

#########################################################################################
#            Predict rent duration and end point for individual rides
#########################################################################################

# Drop samples with missing start station name or end coordinates
# from the original data for individual rides.
data = data.dropna(subset=['start_station_name', 'end_lat', 'end_lng'])

# Array of all station names.
unique_stations = data['start_station_name'].drop_duplicates().values

# Remove samples with negative ride duration.
data = data[data['duration'] > 0]

# Drop samples with too large ride duration (outliers).
threshold = round(data['duration'].quantile(0.97), 2)
print(f'97% quantile for ride duration: {threshold} minutes')
data = data[data['duration'] <= threshold]

# Drop unnecessary columns
data.drop(labels=['same_station', 'ride_id', 'started_at', 'ended_at',
                  'start_station_id', 'end_station_id', 'end_station_name'],
          axis=1, inplace=True)

# Add categorical column with station clusters for start points.
data['cluster'] = data['start_station_name'].apply(lambda x: cluster_dict[x])

# Split the data into train and validation sets.
data_valid = data.sample(frac=0.15, random_state=0)
data_train = data.drop(data_valid.index)

print(f'Train data shape: {data_train.shape}\n'
      f'Validation data shape: {data_valid.shape}')

# Define the target values: one for ride duration
# and two for end point coordinates.
target_columns = ['duration', 'end_lat', 'end_lng']

y_train = data_train[target_columns]
y_valid = data_valid[target_columns]

data_train.drop(labels=target_columns, axis=1, inplace=True)
data_valid.drop(labels=target_columns, axis=1, inplace=True)

tf.keras.backend.clear_session()
del daily_data, data_train_cat, data_valid_cat
gc.collect()

# Process numerical and categorical features separately.
cat_features = ['rideable_type', 'start_station_name', 'member_casual']
num_features = [col for col in data_train.columns if col not in cat_features]

print('Categorical features:', cat_features)
print('Numerical features:', num_features)

data_train_bike = data_train.pop('rideable_type')
data_train_station = data_train.pop('start_station_name')
data_train_user = data_train.pop('member_casual')

data_valid_bike = data_valid.pop('rideable_type')
data_valid_station = data_valid.pop('start_station_name')
data_valid_user = data_valid.pop('member_casual')

data_train = data_train.astype(np.float32)
data_valid = data_valid.astype(np.float32)

# Normalization layer for numerical features.
normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=None)
normalizer.adapt(data_train.sample(n=30_000))  # Adapt to a fraction of train data.

# Three encoding layers for categorical features.
encoding_layer_bike = get_category_encoding_layer(
    data_train_bike.drop_duplicates().values, dtype='string')  # Learn all unique values.
encoding_layer_station = get_category_encoding_layer(unique_stations, dtype='string')
encoding_layer_user = get_category_encoding_layer(
    data_train_user.drop_duplicates().values, dtype='string')

# Model input layers
num_input = tf.keras.Input(shape=(data_train.shape[1],),
                           dtype=tf.float32, name='numeric')

cat_input_bike = tf.keras.Input(shape=(1,), dtype='string', name='bike')
cat_input_station = tf.keras.Input(shape=(1,), dtype='string', name='station')
cat_input_user = tf.keras.Input(shape=(1,), dtype='string', name='user')

# Process inputs separately and combine.
x_1 = normalizer(num_input)
x_2 = encoding_layer_bike(cat_input_bike)
x_3 = encoding_layer_station(cat_input_station)
x_4 = encoding_layer_user(cat_input_user)

x = tf.keras.layers.concatenate([x_1, x_2, x_3, x_4])

x = tf.keras.layers.Dense(
    128, activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.Dense(
    64, activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.Dense(
    32, activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

output = tf.keras.layers.Dense(3)(x)

model = tf.keras.Model([num_input, cat_input_bike, cat_input_station, cat_input_user], output)

model.compile(optimizer='adam',
              loss='mse',
              metrics=[tf.keras.metrics.MeanAbsoluteError(),
                       tf.keras.metrics.RootMeanSquaredError()]
              )

# Visualize the model graph
tf.keras.utils.plot_model(model, show_shapes=True, show_dtype=True)

del data
gc.collect()

history = model.fit(
    x=[data_train, data_train_bike, data_train_station, data_train_user],
    y=y_train, epochs=5, batch_size=1024, shuffle=False)

# Save the model
model.save('tf_user_model')

# Check the magnitude of validation errors and their distribution.
prediction = model.predict(
    x=[data_valid, data_valid_bike, data_valid_station, data_valid_user],
    batch_size=1024)

errors = np.abs(prediction - y_valid.values)

for i, target in enumerate(target_columns):
    print(f'{target} MAE = {np.mean(errors[:, i])}')

    plt.hist(errors[:, i], bins=100)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title(f'Mean Absolute Error: {target}')
    plt.show()
