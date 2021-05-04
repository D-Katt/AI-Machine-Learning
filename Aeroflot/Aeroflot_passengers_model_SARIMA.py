# Модель прогнозирует пассажироперевозки компании "Аэрофлот" на основе ретроспективной динамики.
# Источник данных: https://ir.aeroflot.ru/ru/reporting/traffic-statistics

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

import matplotlib.pyplot as plt

# Считываем данные из экселевского файла:
data = pd.read_excel('Aeroflot_data.xlsx')

# Смотрим на формат данных:
print(data.head())
print(data.shape)
print(data.dtypes)

# Берем для анализа 1 параметр - суммарный объем пассажироперевозок
# (внутренних и международных):
pass_data = data['Passengers carried_thousands']

# Преобразуем индекс в формат datetime с ежемесячной частотой:
pass_data.index = pd.date_range(start='2015-01-01', periods=len(data), freq='M')

# Выводим график:
plt.plot(pass_data)
plt.title('Пассажироперевозки "Аэрофлота"')
plt.xlabel('Периоды')
plt.ylabel('тыс. человек')
plt.tight_layout()
plt.show()

# Декомпозиция данных:
decomposition = seasonal_decompose(pass_data)

observed = decomposition.observed
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Визуализация компонентов:
plt.plot(pass_data.index, observed, label='Исходные данные')
plt.plot(pass_data.index, trend, label='Линия тренда')
plt.plot(pass_data.index, seasonal, label='Сезонные колебания')
plt.plot(pass_data.index, residual, label='Остаточные колебания')
plt.legend()
plt.title('Декомпозиция данных')
plt.tight_layout()
plt.show()

# Графики автокорреляции

plot_acf(pass_data)
plt.title('ACF')
plt.show()

plot_pacf(pass_data)
plt.title('PACF')
plt.show()

# Создаем модель SARIMA:
model_sarima = SARIMAX(pass_data,
                       order=(1, 2, 0),
                       seasonal_order=(1, 2, 0, 12))

decomposition = model_sarima.fit(disp=False)
print(decomposition.summary())

# Визуализируем параметры модели:
decomposition.plot_diagnostics()
plt.show()

# Прогноз на 12 месяцев вперед:
prediction = decomposition.get_forecast(steps=12)

# Доверительные интервалы прогноза:
prediction_int = prediction.conf_int()

# Визуализируем прогноз:
ax = pass_data.plot(label='Исходные данные')
prediction.predicted_mean.plot(ax=ax, label='Прогноз')
ax.fill_between(prediction_int.index,
                prediction_int.iloc[:, 0],
                prediction_int.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Период')
ax.set_ylabel('Пассажироперевозки, тыс. чел.')
plt.legend()
plt.title('Прогноз SARIMA')
plt.show()

# Минимальные и максимальные значения прогноза:
print('\nМин. и макс. значения прогноза SARIMA (тыс. чел.):\n', prediction_int.round(2))

forecast2 = decomposition.forecast(12)
print('\nБазовый прогноз SARIMA (тыс. чел.):\n', forecast2.round(2))

forecast2.plot()
plt.title('Прогноз SARIMA')
plt.ylabel('Пассажироперевозки, тыс. чел.')
plt.show()
