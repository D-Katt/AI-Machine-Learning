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

# Оставляем только столбец с ценой открытия:
data = data['<OPEN>']

# Выводим график с котировками акций:
plt.plot(data)
plt.title('Котировки акций Полиметалла')
plt.ylabel('Цена акции, МосБиржа, руб.')
plt.show()

# Приводим данные к диапазону от 0 до 1:
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))

# Делим данные на учебные и тестовые:
train_data = data_scaled[:-90]  # кроме последних 90 значений
test_data = data_scaled[-90:]  # последние 90 значений

# Создаем генераторы временных рядов
# (при прогнозировании учитываем последние 3 значения ряда):
train_data_gen = TimeseriesGenerator(train_data, train_data,
	length=3, sampling_rate=1, stride=1,
    batch_size=50)

test_data_gen = TimeseriesGenerator(test_data, test_data,
	length=3, sampling_rate=1, stride=1,
	batch_size=10)

# Создаем модель:
model = Sequential([LSTM(4, recurrent_dropout=0.15, return_sequences=True, input_shape=(3, 1)),
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
prediction = scaler.inverse_transform(prediction)

# График с фактическими и прогнозными котировками для тестовых данных:
plt.plot(data[-87:], label='Фактические котировки')
plt.plot(data[-87:].index, prediction.ravel(), label='Прогноз')
plt.title('Котировки акций Полиметалла')
plt.ylabel('Цена акции, МосБиржа, руб.')
plt.legend()
plt.show()
