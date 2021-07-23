"""Model predicts next day's stock price for Polymetall metal company
based on the close price for previous periods. Univariate LSTM model.
Data source: https://www.finam.ru/profile/moex-akcii/polymetal-international-plc/export/
"""

import pandas as pd
import numpy as np
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
LOOKBACK = 3
SAMPLING_RATE = 1
STRIDE = 1

# --------------------------- Functions -----------------------------


def get_data(path: str) -> pd.DataFrame:
    """Function loads stock prices from a local file.
    :param path: Path to a csv file
    :return: DataFrame with close price column and datetime index
    """
    parser = lambda x: datetime.datetime.strptime(x, '%Y%m%d')
    df = pd.read_csv(path, usecols=['<CLOSE>', '<DATE>'],
                     index_col='<DATE>', date_parser=parser)
    display_data(df)
    return df


def display_data(df: pd.DataFrame):
    """Function displays a chart with historical prices.
    :param df: 1-column DataFrame with prices and datetime index
    """
    plt.plot(df)
    plt.title('Stock Price')
    plt.ylabel('Rubles')
    plt.savefig('prices.png')
    plt.show()


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
    plt.savefig('uni_training.png')
    plt.show()


# ------------------------ Data processing --------------------------

# Historical data
data = get_data(FILE_PATH)

# Scale numeric data to the range between 0 and 1
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))

# Create iterables containing stock prices and corresponding next day's prices
input_features = data_scaled[:-1, :]
targets = data_scaled[1:, :]

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
train_ds = tf.keras.preprocessing.sequence.TimeseriesGenerator(
    train_data, train_targets,
    length=LOOKBACK, sampling_rate=SAMPLING_RATE,
    stride=STRIDE, shuffle=True,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.sequence.TimeseriesGenerator(
    val_data, val_targets,
    length=LOOKBACK, sampling_rate=SAMPLING_RATE,
    stride=STRIDE, shuffle=False,
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.preprocessing.sequence.TimeseriesGenerator(
    test_data, test_targets,
    length=LOOKBACK, sampling_rate=SAMPLING_RATE,
    stride=STRIDE, shuffle=False,
    batch_size=BATCH_SIZE
)

# -------------------------------- Model -----------------------------------

# Neural network with Long Short-Term Memory layers
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.LSTM(4, recurrent_dropout=0.15, return_sequences=True,
                             input_shape=(LOOKBACK, 1)),
        tf.keras.layers.LSTM(4, recurrent_dropout=0.15, return_sequences=False),
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
plt.plot(data[-150:], label='Actual data')
plt.plot(data[-120+LOOKBACK-1:-50-1].index, pred_val.ravel(), label='Validation forecast')
plt.plot(data[-50+LOOKBACK-1:-1].index, pred_test.ravel(), label='Test forecast')
plt.ylabel('Rubles')
plt.title('Stock Price')
plt.legend()
plt.savefig('uni_forecast.png')
plt.show()
