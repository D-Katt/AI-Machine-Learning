import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Считываем данные о котировках акций Полиметалла:
data = pd.read_csv('POLY_stock_price.csv')

# Преобразуем столбец с датой в формат datetime:
data['<DATE>'] = pd.to_datetime(data['<DATE>'].astype(str), format='%Y%m%d')

# Преобразуем столбец с датой в индекс:
data.set_index('<DATE>', inplace=True)

# Формируем два массива, содержащие параметры и прогнозируемые значения:
data_X = data[['<OPEN>', '<VOL>']]
data_y = data['<OPEN>']

# Выводим график с котировками акций и объемом торгов:
fig, ax = plt.subplots()
ax_double_x = ax.twinx()

ax.plot_date(data_y.index, data_y.values,
             linestyle='solid', marker=None,
             color='red', label='Цена')
ax.set_ylabel('Цена акции, МосБиржа, руб.')

ax_double_x.plot_date(data_X.index, data_X['<VOL>'].values,
                      linestyle='solid', color='blue',
                      marker=None, label='Объем')
ax_double_x.set_ylabel('Объем торгов')

plt.gcf().autofmt_xdate()
ax.legend(loc='upper left')
ax_double_x.legend(loc='upper right')
ax.set_title('Котировки акций Полиметалла')
fig.tight_layout()
plt.show()

# Приводим данные к диапазону от 0 до 1:
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

data_X_scaled = scaler_X.fit_transform(np.array(data_X).reshape(-1, 2))
data_y_scaled = scaler_y.fit_transform(np.array(data_y).reshape(-1, 1))

# Делим данные на учебные и тестовые:
train_X = data_X_scaled[:-90]
train_y = data_y_scaled[:-90]

test_X = data_X_scaled[-90:]
test_y = data_y_scaled[-90:]

# Создаем генераторы временных рядов (учитываем 3 последних значения):
train_data_gen = TimeseriesGenerator(train_X, train_y,
	length=3, sampling_rate=1, stride=1,
    batch_size=50)

test_data_gen = TimeseriesGenerator(test_X, test_y,
	length=3, sampling_rate=1, stride=1,
	batch_size=10)

# Создаем модель:
model = Sequential([LSTM(4, recurrent_dropout=0.15, return_sequences=True, input_shape=(3, 2)),
                    LSTM(4, recurrent_dropout=0.15, return_sequences=False),
                    Dense(1)])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Обучаем модель (в процессе обучения отслеживаем показатель MSE,
# останавливаем обучение, если нет улучшений для валидационных данных,
# и восстанавливаем веса модели с наилучшим показателем):
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(train_data_gen,
                    epochs=150,
                    verbose=2,
                    validation_data=test_data_gen,
                    callbacks=[early_stop])


def plot_history(histories, key='loss'):
    """Функция создает график с динамикой точности модели
    на учебных и на тестовых данных."""

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Validation')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Training')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    plt.show()


# Выводим график точности в процессе обучения модели:
plot_history([('Model', history)])

# Оцениваем точность модели на тестовых данных:
test_loss = model.evaluate_generator(test_data_gen)
print(f'\nTest loss (MSE): {test_loss}')

# Делаем прогноз для тестовых данных:
prediction = model.predict_generator(test_data_gen)
prediction = scaler_y.inverse_transform(prediction)

# График с фактическими и прогнозными котировками для тестовых данных:
plt.plot_date(data_y[-87:].index, data_y[-87:].values,
              linestyle='solid', marker=None, label='Фактические котировки')
plt.plot_date(data_y[-87:].index, prediction.ravel(),
              linestyle='solid', marker=None, label='Прогноз')
plt.gcf().autofmt_xdate()
plt.title('Котировки акций Полиметалла')
plt.ylabel('Цена акции, МосБиржа, руб.')
plt.legend()
plt.show()
