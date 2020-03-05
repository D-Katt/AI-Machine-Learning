# Разработка модели, делающей прогноз оборота розничной торговли в России.
# Используется модель LSTM.

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

from collections import deque

import warnings
warnings.filterwarnings('ignore')

# Считываем данные по ежемесячному обороту розничной торговли:
data_m = pd.read_excel('retail_data_monthly.xlsx')

# Оставляем один столбец с прогнозируемым показателем:
data_m = data_m['Retail_turnover']

# Создаем индекс в формате datetime с ежемесячной частотой:
data_m.index = pd.date_range(start='2000-01-31', periods=len(data_m), freq='M')

# Выводим график по всем имеющимся данным:
plt.plot(data_m)
plt.title('Оборот розничной торговли, млн. руб.')
plt.show()

# Сужаем временной период:
data_m = data_m[data_m.index > pd.to_datetime('2014-03-31')]

# Для большей наглядности преобразуем млн. в млрд.:
data_m = data_m / 1_000

# Выводим график:
plt.plot(data_m)
plt.title('Оборот розничной торговли, млрд. руб.')
plt.show()

# Преобразуем данные к диапазону от 0 до 1:
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(np.array(data_m).reshape(-1, 1))

# Делим данные на учебные и тестовые:
division = -24
train_data = data_scaled[:division]  # кроме последних 24 значений
test_data = data_scaled[division:]  # последние 24 значения

# Создаем генераторы временных рядов
# (при прогнозировании учитываем последние 12 значений ряда):
look_back = 12

train_data_gen = TimeseriesGenerator(train_data, train_data,
	length=look_back, sampling_rate=1, stride=1,
    batch_size=4)

test_data_gen = TimeseriesGenerator(test_data, test_data,
	length=look_back, sampling_rate=1, stride=1,
	batch_size=4)

# Создаем модель:
model = Sequential([LSTM(250, activation='relu',
                         recurrent_dropout=0.15,
                         kernel_regularizer=regularizers.l2(0.01),
                         return_sequences=True,
                         input_shape=(look_back, 1)),
                    LSTM(250, activation='relu',
                         recurrent_dropout=0.15,
                         kernel_regularizer=regularizers.l2(0.01),
                         return_sequences=False),
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
    plt.figure(figsize=(16, 10))

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

# Если модель показала высокую точность на тестовых данных,
# сохраняем ее для последующего прогнозирования:
if test_loss < 0.01:
    model.save('best_model.h5')
else:
    exit()

# Если полученная модель демонстрирует недостаточно высокую точность,
# код ниже не выполняется.

# Делаем прогноз для тестовых данных:
prediction = model.predict_generator(test_data_gen)
prediction = scaler.inverse_transform(prediction)

# График с фактическими и прогнозными данными:
plt.plot(data_m, label='Фактические данные')
plt.plot(data_m[division+look_back:].index, prediction.ravel(), label='Прогноз')
plt.title('Оборот розничной торговли')
plt.ylabel('млрд. руб.')
plt.legend()
plt.tight_layout()
plt.savefig('validation.png')
plt.show()

# Делаем прогноз на 12 месяцев вперед на основе последних доступных данных.

# Создаем очередь из последних 12 значений Time Series:
last_data = deque(data_scaled[-12:])

forecast = []  # Список для добавления значений прогноза

for i in range(12):
    # Преобразуем последние 12 значений в трехмерный np.array:
    model_input = np.array(last_data, dtype='float32')
    model_input = model_input.reshape((1, 12, 1))
    # Делаем на основе этих данных прогноз на следующий месяц:
    next_month = model.predict(model_input)
    # Добавляем прогнозное значение в конец очереди и удаляем 1-й элемент очереди:
    last_data.append(next_month)
    last_data.popleft()
    # Преобразуем прогнозное значение из диапазона 0-1 к исходному масштабу:
    next_month = scaler.inverse_transform(next_month)
    # Добавляем это прогнозное значение в список с прогнозом:
    forecast.append(next_month.ravel())

# График с динамикой показателя и прогнозом на 12 месяцев вперед:
plt.plot(data_m, label='Фактические данные')
plt.plot(pd.date_range(start='2020-02-01', periods=12, freq='M'), forecast, label='Прогноз')
plt.title('Оборот розничной торговли')
plt.ylabel('млрд. руб.')
plt.legend()
plt.savefig('forecast.png')
plt.show()
