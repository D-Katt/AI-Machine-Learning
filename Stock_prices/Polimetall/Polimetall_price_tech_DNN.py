"""Model predicts next day's stock price for Polymetall metal company
based on the close price and technical indicators for the previous period.
Densely connected neural network.
Data source: https://www.finam.ru/profile/moex-akcii/polymetal-international-plc/export/
"""

import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Plots display settings
plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams.update({'font.size': 14})

FILE_PATH = 'POLY_stock_price.csv'

# Tensorflow settings
EPOCHS = 1000
PATIENCE = 5
BATCH_SIZE = 64

# --------------------------- Functions -----------------------------


def get_data(path: str) -> pd.DataFrame:
    """Function loads stock prices from a local file.
    :param path: Path to a csv file
    :return: DataFrame with close price column and datetime index
    """
    parser = lambda x: datetime.datetime.strptime(x, '%Y%m%d')
    df = pd.read_csv(path, usecols=['<CLOSE>', '<DATE>'],
                     index_col='<DATE>', date_parser=parser)
    return df


def daily_returns(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Function computes daily return values for selected parameter.
    :param df: DataFrame with original values
    :param column: Name of the column with selected parameter
    :return: Updated DataFrame
    """
    df[f'{column}_returns'] = df[column] / df[column].shift(1) - 1
    return df


def calculate_macd(df: pd.DataFrame, column: str,
                   nslow: int = 26, nfast: int = 12) -> pd.DataFrame:
    """Function computes moving average convergence-divergence (MACD)
    indicator for price values from selected 'column' in 'df'.
    :param df: DataFrame with original values
    :param column: Name of the column to use
    :param nslow: Larger time span
    :param nfast: Shorter time span
    :return: Updated DataFrame containing difference and MACD values
    """
    # Difference between two exponential moving averages
    # to measure momentum in a security
    emaslow = df[column].ewm(
        span=nslow, min_periods=nslow, adjust=True, ignore_na=False
    ).mean()
    emafast = df[column].ewm(
        span=nfast, min_periods=nfast, adjust=True, ignore_na=False
    ).mean()
    df[f'dif_{column}'] = emafast - emaslow
    # 9 days MACD indicator
    df[f'macd_{column}'] = df[f'dif_{column}'].ewm(
        span=9, min_periods=9, adjust=True, ignore_na=False
    ).mean()
    return df


def calculate_rsi(df: pd.DataFrame, column: str,
                  periods: int = 14) -> pd.DataFrame:
    """Function computes Relative Strength Index (RSI)
    for price values from selected 'column' in 'df'.
    :param df: DataFrame with original values
    :param column: Name of the column to use
    :param periods: Number of days
    :return: Updated DataFrame with RSI values
    """
    # Price difference with the previous day
    delta = df[column].diff()

    # Gain and loss
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    avg_gain = up.ewm(com=periods, adjust=False).mean()
    avg_loss = down.ewm(com=periods, adjust=False).mean().abs()

    df[f'rsi_{column}'] = 100 - 100 / (1 + avg_gain / avg_loss)
    return df


def calculate_sma(df: pd.DataFrame, column: str,
                  periods: int = 15) -> pd.Series:
    """Function computes Simple Moving Average (SMA)
    for price values from selected 'column' in 'df'.
    :param df: DataFrame with original values
    :param column: Name of the column to use
    :param periods: Number of days
    :return: Series with SMA values
    """
    return df[column].rolling(window=periods, min_periods=periods, center=False).mean()


def calculate_bands(df: pd.DataFrame, column: str,
                    peroids: int = 15) -> pd.DataFrame:
    """Function calculates Bollinger Bands
    for price values from selected 'column' in 'df'.
    :param df: DataFrame with original values
    :param column: Name of the column to use
    :param peroids: Number of days
    :return: Updated DataFrame containing upper and lower band values
    """
    std = df[column].rolling(window=peroids, min_periods=peroids, center=False).std()
    sma = calculate_sma(df, column)
    df[f'upper_band_{column}'] = sma + (2 * std)
    df[f'lower_band_{column}'] = sma - (2 * std)
    return df


def plot_history(hist):
    """Function plots a chart with training and validation metrics.
    :param hist: Tensorflow history object from model.fit()
    """
    # Losses
    mae = hist.history['loss']
    val_mae = hist.history['val_loss']

    # Epochs to plot along x axis
    x_axis = range(1, len(mae) + 1)

    plt.plot(x_axis, mae, 'bo', label='Training')
    plt.plot(x_axis, val_mae, 'ro', label='Validation')
    plt.title('Mean Squared Error')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig('DNN_training.png')
    plt.show()


# ------------------------ Data processing --------------------------

# Historical data
data = get_data(FILE_PATH)

# Calculate daily returns for close price
data = daily_returns(data, '<CLOSE>')

# Add technical indicators
data = calculate_rsi(data, '<CLOSE>')
data = calculate_bands(data, '<CLOSE>')
data = calculate_macd(data, '<CLOSE>')

# Remove rows with missing values for technical indicators
# (where rolling functions produced NaNs)
data.dropna(inplace=True)

# Create iterables containing input features
# and corresponding next day's prices
input_features = data.iloc[:-1, :].values
targets = data.iloc[1:, 0].values.reshape(-1, 1)

# Scale down predicted values for better model convergence
scaler = MinMaxScaler()
targets = scaler.fit_transform(targets)

# Leave latest periods for test and validation purposes
train_data = input_features[:-120]
val_data = input_features[-120:-50]
test_data = input_features[-50:]

train_targets = targets[:-120]
val_targets = targets[-120:-50]
test_targets = targets[-50:]

print(f'Dataset shape: {data.shape}')
print(f'Train data: {train_data.shape}')
print(f'Validation data: {val_data.shape}')
print(f'Test data: {test_data.shape}')

# Create tensorflow dataset objects
train_ds = tf.data.Dataset.from_tensor_slices(
    (train_data, train_targets))\
    .shuffle(buffer_size=len(train_data))\
    .batch(BATCH_SIZE)

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_data, val_targets)).batch(BATCH_SIZE)

test_ds = tf.data.Dataset.from_tensor_slices(
    (test_data, test_targets)).batch(BATCH_SIZE)

# -------------------------------- Model -----------------------------------

# Normalization layer to scale numeric input data
normalizer = tf.keras.layers.experimental.preprocessing.Normalization(
    input_shape=(7,)
)
normalizer.adapt(train_data)

# Densely connected neural network
model = tf.keras.models.Sequential(
    [
        normalizer,
        tf.keras.layers.Dense(
            32, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(
            16, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(1)
    ]
)

model.compile(optimizer='adam', loss='mse',
              metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
model.summary()

# Train the model until validation accuracy stops improving
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=PATIENCE, restore_best_weights=True
)

history = model.fit(train_ds,
                    epochs=EPOCHS,
                    verbose=2,
                    validation_data=val_ds,
                    callbacks=[early_stop])

plot_history(history)

# Evaluate the model on the test set
test_loss, test_mape = model.evaluate(test_ds)
print(f'MSE loss on test data: {test_loss}\nMAPE: {test_mape}')

# Forecasts for validation and test periods
pred_val = model.predict(val_ds)
pred_val = scaler.inverse_transform(pred_val)
pred_test = model.predict(test_ds)
pred_test = scaler.inverse_transform(pred_test)

# Visualize forecast vs. actual prices
plt.plot(data[-150:]['<CLOSE>'], label='Actual data')
plt.plot(data[-120-1:-50-1].index, pred_val.ravel(), label='Validation forecast')
plt.plot(data[-50-1:-1].index, pred_test.ravel(), label='Test forecast')
plt.ylabel('Rubles')
plt.title('Stock Price')
plt.legend()
plt.savefig('DNN_forecast.png')
plt.show()
