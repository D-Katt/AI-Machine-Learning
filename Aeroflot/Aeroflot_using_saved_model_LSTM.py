import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from collections import deque

# Считываем данные:
data = pd.read_excel('Aeroflot_data.xlsx')

# Берем для анализа 1 параметр - суммарный объем пассажироперевозок
# (внутренних и международных):
pass_data = data['Passengers carried_thousands']

# Преобразуем индекс в формат datetime с ежемесячной частотой:
pass_data.index = pd.date_range(start='2015-01-01', periods=len(data), freq='M')

# Приводим данные к диапазону от 0 до 1:
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(np.array(pass_data).reshape(-1, 1))

# Делим данные на учебные и тестовые:
division = -24
train_data = data_scaled[:division]  # кроме последних 24 значений
test_data = data_scaled[division:]  # последние 24 значения

# Создаем генераторы временных рядов
# (при прогнозировании учитываем последние 12 значений ряда):
look_back = 12

train_data_gen = TimeseriesGenerator(train_data, train_data,
	length=look_back, sampling_rate=1, stride=1,
    batch_size=3)

test_data_gen = TimeseriesGenerator(test_data, test_data,
	length=look_back, sampling_rate=1, stride=1,
	batch_size=3)

# Восстанавливаем из файла сохраненную модель, включая веса и оптимизатор:
best_model = load_model('best_model.h5')

# Архитектура модели:
best_model.summary()

# Оцениваем точность модели на тестовых данных:
test_loss = best_model.evaluate_generator(test_data_gen)
print(f'\nTest loss (MSE): {test_loss}')

# Делаем прогноз для тестовых данных:
prediction = best_model.predict_generator(test_data_gen)
prediction = scaler.inverse_transform(prediction)

# График с фактическими и прогнозными значениями для тестовых данных:
plt.plot(pass_data, label='Фактические данные')
plt.plot(pass_data[division+look_back:].index, prediction.ravel(), label='Прогноз')
plt.title('Пассажироперевозки "Аэрофлота"')
plt.xlabel('Периоды')
plt.ylabel('тыс. человек')
plt.legend()
plt.savefig('validation.png')
plt.show()

# Делаем прогноз на 12 месяцев вперед на основе последних доступных данных.

# Создаем очередь из последних 12 значений Time Series:
last_data = deque(data_scaled[-12:])

forecast = []  # Список для добавления значений прогноза

for i in range(12):
    # Преобразуем последние 12 значений в трехмерный np.array:
    model_input = np.array(last_data, dtype='float32')
    model_input = model_input.reshape((1, look_back, 1))
    # Делаем на основе этих данных прогноз на следующий месяц:
    next_month = best_model.predict(model_input)
    # Добавляем прогнозное значение в конец очереди и удаляем 1-й элемент очереди:
    last_data.append(next_month)
    last_data.popleft()
    # Преобразуем прогнозное значение из диапазона 0-1 к исходному масштабу:
    next_month = scaler.inverse_transform(next_month)
    # Добавляем это прогнозное значение в список с прогнозом:
    forecast.append(next_month.ravel())

# График с динамикой показателя и прогнозом на 12 месяцев вперед:
plt.plot(pass_data, label='Фактические данные')
plt.plot(pd.date_range(start='2020-02-01', periods=12, freq='M'), forecast, label='Прогноз')
plt.title('Пассажироперевозки "Аэрофлота"')
plt.xlabel('Периоды')
plt.ylabel('тыс. человек')
plt.legend()
plt.savefig('forecast.png')
plt.show()
