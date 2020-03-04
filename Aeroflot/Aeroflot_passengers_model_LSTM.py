import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from collections import deque

# Считываем данные из экселевского файла:
data = pd.read_excel('Aeroflot_data.xlsx')

# Берем для анализа 1 параметр - суммарный объем пассажироперевозок
# (внутренних и международных):
pass_data = data['Passengers carried_thousands']

# Преобразуем индекс в формат datetime с ежемесячной частотой:
pass_data.index = pd.date_range(start='2015-01-01', periods=len(data), freq='M')

# Выводим график:
plt.plot_date(pass_data.index, pass_data.values, linestyle='solid', marker=None)
plt.gcf().autofmt_xdate()
plt.title('Пассажироперевозки "Аэрофлота"')
plt.xlabel('Периоды')
plt.ylabel('тыс. человек')
plt.tight_layout()
plt.show()

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

# График с фактическими и прогнозными значениями для тестовых данных:
plt.plot(pass_data, label='Фактические данные')
plt.plot(pass_data[division+look_back:].index, prediction.ravel(), label='Прогноз')
plt.title('Пассажироперевозки "Аэрофлота"')
plt.xlabel('Периоды')
plt.ylabel('тыс. человек')
plt.legend()
plt.show()
plt.savefig('validation.png')

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
plt.plot(pass_data, label='Фактические данные')
plt.plot(pd.date_range(start='2020-02-01', periods=12, freq='M'), forecast, label='Прогноз')
plt.title('Пассажироперевозки "Аэрофлота"')
plt.xlabel('Периоды')
plt.ylabel('тыс. человек')
plt.legend()
plt.show()
plt.savefig('forecast.png')
