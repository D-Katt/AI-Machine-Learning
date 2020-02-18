# Прогнозирование Time Series на основе данных о количестве иностранных туристов.

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6

# Считываем поквартальные данные о количестве туристов
# (данные кумулятивные - с начала года до указанной даты):
data = pd.read_csv('foreign_tourists.csv', usecols=['Date', 'Foreign_tourists'])
data.Date = pd.to_datetime(data.Date)

# Преобразуем столбец с датами в индекс:
data = data.set_index(['Date'])

# Выводим график с динамикой туристического потока,
# чтобы визуально убедиться, что данные коррелируют с временными интервалами:
plt.plot(data)
plt.xlabel('Дата')
plt.ylabel('Кол-во туристов с начала года')
plt.title('Иностранные туристы в России')
plt.tight_layout()
plt.show()

# Вычисляем скользящее среднее и стандартное отклонение
# для окна в 4 элемента Time Series (4 квартала):
rolmean = data.rolling(window=4).mean()
rolstd = data.rolling(window=4).std()

# Выводим график со статистическими показателями:
plt.plot(data, label='Исходные данные')
plt.plot(rolmean, label='Rolling Mean')
plt.plot(rolstd, label='Rolling Std')
plt.legend()
plt.title('Скользящее среднее и стандартное отклонение')
plt.tight_layout()
plt.show()

# Dickey-Fuller test:
print('Результат Dickey-Fuller test:')
dftest = adfuller(data['Foreign_tourists'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test_statistic', 'p-value', 'Lags_used', 'Number_of_observations'])

for key, value in dftest[4].items():
    dfoutput['Critical_value(%s)' % key] = value

print(dfoutput)

# Визуально оцениваем тренд для логарифма исходных данных:
data_logScale = np.log(data)
plt.plot(data_logScale)
plt.title('Логарифм ряда исходных данных')
plt.show()

movingAverage = data_logScale.rolling(window=4).mean()
movingStd = data_logScale.rolling(window=4).std()
plt.plot(data_logScale, label='Лог. данных')
plt.plot(movingAverage, label='Rolling Mean')
plt.title('Скользящее среднее и станд. отклонение для логарифма данных')
plt.show()

dataLogScaleMinusMovingAverage = data_logScale - movingAverage
print(dataLogScaleMinusMovingAverage)

# Удаляем строки с отсутствующими данными:
dataLogScaleMinusMovingAverage.dropna(inplace=True)
print(dataLogScaleMinusMovingAverage)


def test_stationary(timeseries):
    """Функция принимает объект Time Series, вычисляет показатели
    скользящего среднего и стандартного отклонения и выводит график,
    выполняет Dickey-Fuller test и выводит его показатели."""

    # Вычисляем показатели:
    movingAverage = timeseries.rolling(window=4).mean()
    movingSTD = timeseries.rolling(window=4).std()

    # Выводим график:
    plt.plot(timeseries, label='Original')
    plt.plot(movingAverage, label='Rolling Mean')
    plt.plot(movingSTD, label='Rolling Std')
    plt.legend()
    plt.title('Скользящее среднее и стандартное отклонение')
    plt.tight_layout()
    plt.show()

    # Dickey-Fuller test:
    print('Результат Dickey-Fuller test:')
    dftest = adfuller(timeseries.values, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test_statistic', 'p-value', 'Lags_used', 'Number_of_observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical_value(%s)' % key] = value
    print(dfoutput)


# Вызываем функцию:
test_stationary(dataLogScaleMinusMovingAverage)

# Средневзвешенный показатель:
exponentialDecayWeightedAverage = data_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(data_logScale)
plt.plot(exponentialDecayWeightedAverage)
plt.title('Средневзвешенный показатель для логарифма данных')

datasetLogScaleMinusMovingExponentialDecayAverage = data_logScale - exponentialDecayWeightedAverage
test_stationary(datasetLogScaleMinusMovingExponentialDecayAverage)

# Сдвигаем данные:
datasetLogDiffShifting = data_logScale - data_logScale.shift()
plt.plot(datasetLogDiffShifting)

# Удаляем отсутствующие значения:
datasetLogDiffShifting.dropna(inplace=True)
test_stationary(datasetLogDiffShifting)

# Компоненты Time Series:
decomposition = seasonal_decompose(data_logScale)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.plot(data_logScale, label='Original log')
plt.plot(trend, label='Trend')
plt.plot(seasonal, label='Seasonality')
plt.plot(residual, label='Residuals')
plt.legend()
plt.tight_layout()
plt.show()

decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationary(decomposedLogData)

lag_acf = acf(datasetLogDiffShifting, nlags=20)
lag_pacf = pacf(datasetLogDiffShifting, nlags=20, method='ols')
# 'ols' - ordinary least squares method.

# Выводим график ACF:
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')
plt.show()

# Выводим график PACF:
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()

# AR Model:
model = ARIMA(data_logScale, order=(2, 1, 0))  # (P, Q, d) values
results_AR = model.fit()
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f' % sum((results_AR.fittedvalues - datasetLogDiffShifting['Foreign_tourists']) ** 2))
# RSS - Residual Sum of Squares (чем ниже этот показатель - тем лучше)
plt.show()
print('Plotting AR Model')

# MA Model:
model = ARIMA(data_logScale, order=(0, 1, 0))
results_MA = model.fit()
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f' % sum((results_MA.fittedvalues - datasetLogDiffShifting['Foreign_tourists']) ** 2))
plt.show()
print('Plotting MA Model')

# ARIMA Model:
model = ARIMA(data_logScale, order=(2, 1, 0))
results_ARIMA = model.fit()

prediction_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(prediction_ARIMA_diff.head())

# Нарастающим итогом:
prediction_ARIMA_diff_cumsum = prediction_ARIMA_diff.cumsum()
print(prediction_ARIMA_diff_cumsum.head())

# Делаем прогноз:
prediction_ARIMA_log = pd.Series(data_logScale['Foreign_tourists'].iloc[0], index=data_logScale.index)
prediction_ARIMA_log = prediction_ARIMA_log.add(prediction_ARIMA_diff_cumsum, fill_value=0)
print(prediction_ARIMA_log.head())

# Возвращаем данные в исходный масштаб:
prediction_ARIMA = np.exp(prediction_ARIMA_log)
plt.plot(data, label='Original')
plt.plot(prediction_ARIMA, label='ARIMA Prediction')
plt.legend()
plt.show()

print(data_logScale)

# Делаем прогноз на 3 года:

results_ARIMA.plot_predict(1, 34)  # Выводим график с доверительным интервалом

x = results_ARIMA.forecast(steps=12)  # Прогноз на 3 года, т.е. на 3 * 4 кварталов
print(x)
