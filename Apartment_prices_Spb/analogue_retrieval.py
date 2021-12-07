"""Apartment evaluation algorithm based on querying listings
parsed from popular web sites.
Nearest Neighbours models are trained for all apartment types
in the selected region. A portion of the data for each type is used for testing.
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

import random
import joblib
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

PATH = '../input/russia-real-estate-20182021/all_v2.csv'

MIN_AREA = 20  # Outlier range for floor area
MAX_AREA = 200

MIN_KITCHEN = 6  # Outlier range for kitchen area
MAX_KITCHEN = 30

MIN_PRICE = 1_500_000  # Outlier range for price
MAX_PRICE = 50_000_000

MIN_SQM_PRICE = 75_000  # Outlier range for price per sq. meter
MAX_SQM_PRICE = 250_000

TEST_SIZE = 0.1

# Features to use in Nearest Neighbours model.
FEATURES = ['geo_lat', 'geo_lon', 'building_type', 'level', 'levels',
            'area', 'kitchen_area', 'object_type', 'year', 'month',
            'level_to_levels', 'area_to_rooms']

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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Function updates DataFrame adding temporal features
    and ratios for "area" and "level" parameters.
    :param df: Original DataFrame
    :return: Updated DataFrame
    """
    df['date'] = pd.to_datetime(df['date'])
    # Replace "date" with numeric features for year and month.
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df.drop(['date', 'time'], axis=1, inplace=True)
    # Apartment floor in relation to total number of floors.
    df['level_to_levels'] = df['level'] / df['levels']
    # Average size of room in the apartment.
    df['area_to_rooms'] = (df['area'] / df['rooms']).abs()
    # Fix division by zero.
    df.loc[df['area_to_rooms'] == np.inf, 'area_to_rooms'] = \
        df.loc[df['area_to_rooms'] == np.inf, 'area']
    return df


def select_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Function selects the listings belonging to a specified region.
    :param df: Original DataFrame with all listings
    :param region: Region ID
    :return: Filtered DataFrame
    """
    df = df[df['region'] == region]
    return df.drop('region', axis=1)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Function removes outliers from the DataFrame.
    :param df: Original DataFrame
    :return: Updated DataFrame
    """
    # Column actually contains -1 and -2 values presumably for studio apartments.
    df['rooms'] = df['rooms'].apply(lambda x: 0 if x < 0 else x)
    df['price'] = df['price'].abs()  # Fix negative values
    # Drop price and area outliers.
    df = df[(df['area'] <= MAX_AREA) & (df['area'] >= MIN_AREA)]
    df = df[(df['price'] <= MAX_PRICE) & (df['price'] >= MIN_PRICE)]
    # Drop outliers based on price per sq. meter.
    df['sqm_price'] = df['price'] / df['area']
    df = df[(df['sqm_price'] >= MIN_SQM_PRICE) & (df['sqm_price'] <= MAX_SQM_PRICE)]
    # Fix kitchen area outliers.
    # At first, replace all outliers with 0.
    df.loc[(df['kitchen_area'] >= MAX_KITCHEN) | (df['area'] <= MIN_AREA), 'kitchen_area'] = 0
    # Then calculate kitchen area based on the floor area, except for studios.
    erea_mean, kitchen_mean = df[['area', 'kitchen_area']].quantile(0.5)
    kitchen_share = kitchen_mean / erea_mean
    df.loc[(df['kitchen_area'] == 0) & (df['rooms'] != 0), 'kitchen_area'] = \
        df.loc[(df['kitchen_area'] == 0) & (df['rooms'] != 0), 'area'] * kitchen_share

    return df


# ------------------------- Data Processing & Analysis-------------------------------

set_display()

data = pd.read_csv(PATH)
print(f'Data shape: {data.shape}')
print(data.head())

print(data.info())
print(data.describe())

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

# Regions are encoded with numeric IDs.
regions = data['region'].value_counts()

plt.hist(regions.values, bins=10)
plt.title('Listings by Region')
plt.show()

# Select random region with at least 100,000 apartments (before data cleaning)
# to test the algorithm on samples from one geographical area.
regions = regions[regions >= 100_000].index
region = random.choice(regions)
print(f'Selected region ID: {region}')

# Drop other regions and outliers, generate new features.
data = select_region(data, region).pipe(clean_data).pipe(add_features)
print(f'Data shape: {data.shape}')

# Basic statistics for the target value.
mean_price = int(data['price'].mean())
median_price = int(data['price'].median())

std = int(data['price'].std())

min_price = int(data['price'].min())
max_price = int(data['price'].max())

print(f'Price range: {min_price} - {max_price}')
print(f'Mean price: {mean_price}\nMedian price: {median_price}')
print(f'Standard deviation: {std}')

plt.hist(data['price'], bins=20)
plt.axvline(mean_price, label='Mean Price', color='red')
plt.axvline(median_price, label='Median Price', color='green')
plt.legend()
plt.xlabel('Apartment Price, Rubles')
plt.title('Price Distribution')
plt.show()

correlation = data.corr()
ax = sns.heatmap(correlation, center=0, cmap='RdBu_r')
l, r = ax.get_ylim()
ax.set_ylim(l + 0.5, r - 0.5)
plt.yticks(rotation=0)
plt.title('Correlation Matrix')
plt.show()

# To keep track of MAE and MSE for all apartment types.
all_errors = []
all_mse = []

# Create a separate model for each apartment type
# (studio, 1-room, 2-room, etc.).
for n_rooms in data['rooms'].unique():
    print('-' * 50)
    print(f'Creating model for {n_rooms} rooms apartments.')
    subset = data[data['rooms'] == n_rooms].copy()

    # Select a random subset of data to use as queries.
    n_test_samples = int(len(subset) * TEST_SIZE)
    test_queries = subset.sample(n=n_test_samples)
    subset = subset.drop(test_queries.index)
    print(f'Train subset data shape: {subset.shape}')

    # Check the size of the subset.
    n_train_samples = len(subset)

    if n_train_samples < 10:
        print('Warning: Not enough samples for this apartment type.')
        continue

    # Set number of neighbours to 10.
    n_neighbours = 10

    # Create a pipeline including scaler and neighbour search model.
    pipe = make_pipeline(
        StandardScaler(),
        NearestNeighbors(n_neighbors=n_neighbours, radius=0.3, n_jobs=-1)
    )

    pipe.fit(subset[FEATURES])

    # Save the pipeline using region ID and number of rooms to define a filename.
    joblib.dump(value=pipe, filename=f'region{region}_rooms{n_rooms}.joblib')

    # Select the most similar apartments for all test samples.
    query = pipe[0].transform(test_queries[FEATURES])  # Scale input features.
    similar_idx = pipe[1].kneighbors(query, n_neighbours, return_distance=False)


    def median_price(idx: np.array):
        return subset.iloc[idx, :]['price'].median()


    # Transform indexes of similar apartments into median prices.
    y_pred = np.apply_along_axis(median_price, 1, similar_idx)

    # Compare with the actual price.
    errors = np.abs(test_queries['price'].values - y_pred).tolist()

    mse = [err ** 2 for err in errors]
    print(f'Average MAE for {n_rooms} rooms apartment: {int(np.mean(errors))}')
    print(f'Average RMSE for {n_rooms} rooms apartment: {int(np.sqrt(np.mean(mse)))}')

    all_errors.extend(errors)
    all_mse.extend(mse)

mae = int(np.mean(all_errors))
rmse = int(np.sqrt(np.mean(all_mse)))

print(f'Average MAE for all queries: {mae}')
print(f'Average RMSE for all queries: {rmse}')
print(f'MAE / std ratio: {mae / std}')

plt.hist(all_errors, bins=20)
plt.axvline(std, label='Standard Deviation', color='red')
plt.legend()
plt.title('Price MAE')
plt.show()
