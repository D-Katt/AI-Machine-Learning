"""Module uses FB Prophet model to predict monthly passenger turnover
and identify anomalies in the time series
based on the statistics of Aeroflot airline company.
"""

from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams.update({'font.size': 14})

# Ссылка на исходные данные
DATA_LINK = r'https://ir.aeroflot.ru/fileadmin/user_upload/files/eng/companys_reporting/Operating_highlights/2021/traffic_figures_03_2021_eng.xls'

# Количество прогнозных периодов для модели
N_PERIODS = 12

# --------------------------------- Функции ----------------------------------


def load_operational_stats(url: str) -> pd.DataFrame:
    """Function loads data via URL and selects operational parameters.
    :param url: URL to data source in Excel format
    :return: DataFrame with operational data
    """
    data = pd.read_excel(url, header=1)
    # Верхняя часть таблицы - ежемесячный трафик в абсолютном выражении
    data = data.head(39)
    # Убираем пустые строки и столбцы
    data = data.loc[2:, 'Main operating indicators':]
    return data


def select_parameter(df: pd.DataFrame, par: str) -> pd.DataFrame:
    """Function selects a single parameter from operational data
    and returns a cleaned DataFrame with columns=['ds', 'y'].
    :param df: Original DataFrame with multiple parameters
    :param par: Selected parameter
    :return: pd.DataFrame with time steps and selected parameter values
    """
    # Выбираем строку с данными по указанному параметру
    selected_data = df[df['Main operating indicators'] == par].copy()
    # Убираем столбец с ед. изм.
    selected_data.drop('Unnamed: 3', axis='columns', inplace=True)
    # Заменяем перенос строки в названиях столбцов пробелом
    selected_data.columns = selected_data.columns.str.replace('\n', ' ')
    # Преобразуем таблицу в вертикальный формат
    selected_data = pd.melt(selected_data, id_vars=['Main operating indicators'],
                            var_name='ds', value_name='y')
    # Преобразуем периоды в формат datetime с ежемесячной частотой
    selected_data['ds'] = pd.date_range(start='2014-01-01', periods=len(selected_data), freq='M')
    # Убираем лишние столбцы
    selected_data.drop('Main operating indicators', axis='columns', inplace=True)
    return selected_data


def get_forecast(data: pd.DataFrame, n_periods) -> tuple:
    """Function initializes FB Prophet model and trains it,
    produces a monthly forecast for 'n_periods' ahead.
    :param data: DataFrame with time steps and parameter values
    :param n_periods: Number of periods (months) to be forecasted
    :return: Tuple with fitted model and DataFrame with the forecast
    """
    model = Prophet(seasonality_mode='multiplicative', mcmc_samples=300)
    model.fit(data)
    # Временная шкала для прогнозируемого периода
    future = model.make_future_dataframe(periods=n_periods, freq='M')
    forecast = model.predict(future)

    return model, forecast


def calculate_errors(df_retrospect: pd.DataFrame,
                     df_forecast: pd.DataFrame,
                     n_periods: int) -> pd.DataFrame:
    """Function adds predicted values from 'df_forecast' to 'df_retrospect'
    and calculates absolute and relative difference with actual values
    for all past periods.
    :param df_retrospect: DataFrame with retrospective data
    :param df_forecast: DataFrame with predicted parameters returned by FB Prophet model
    :param n_periods: Number of future periods included in 'df_forecast'
    :return: Updated DataFrame with retrospective data
    """
    # Расчетные значения модели без прогнозного периода
    df_retrospect['forecast'] = df_forecast['yhat'].head(len(df_forecast) - n_periods).values
    # Разность прогноза и фактического значения
    df_retrospect['error_abs'] = df_retrospect['forecast'] - df_retrospect['y']
    # Ошибка прогноза в долях от 1
    df_retrospect['error_rel'] = df_retrospect['forecast'] / df_retrospect['y'] - 1
    return df_retrospect


def errors_distribution(df: pd.DataFrame, par: str) -> tuple:
    """Function displays a histogram or absolute or relative errors
    of FB Prophet based on retrospective data.
    :param df: DataFrame with actual data with columns=['ds', 'y']
    :param par: Parameter defining type of errors to be displayed
    (one of 'error_abs' or 'errors_rel')
    :return Tuple of float values containing mean errors and standard deviation for errors
    """
    mean_error = np.round(df[par].mean(), 2)
    std_error = np.round(df[par].std(), 2)
    print(f'Среднее расхождение: {mean_error}\nСтандартное отклонение: {std_error}')

    units = 'тыс. чел.' if par == 'error_abs' else 'в долях от 1'

    df[par].hist(bins=30)
    plt.xlabel(f'Ошибка, {units}')
    plt.ylabel('Частота')
    plt.figtext(0.7, 0.8, f'mean={mean_error}\nstd={std_error}', fontweight="bold")
    plt.title('Расхождение модели с фактическими данными')

    return mean_error, std_error


def plot_anomalies(df: pd.DataFrame, periods: np.array):
    """Function displays actual data with marked anomalous periods.
    :param df: DataFrame with retrospective data and time steps
    :param periods: Array of dates
    """
    max_val = df['y'].max()
    plt.plot(df['ds'], df['y'])
    plt.vlines(periods, ymin=0, ymax=max_val, colors='r', linestyles='dashed')
    plt.ylabel('тыс. человек')
    plt.title('Пассажироперевозки "Аэрофлота"')


def find_anomalous_days(df: pd.DataFrame,
                        par: str,
                        min_threshold: float,
                        max_threshold: float) -> np.array:
    """Function selects anomalous dates based on the threshold of difference
    between values modelled by FB Prophet model and actual values for past periods.
    :param df: DataFrame containing retrospective data, model estimations
    and differences between the modelled values and ground truth
    :param par: Type of difference to be used (one of 'error_abs' or 'error_rel')
    :param min_threshold: Lower boundary for errors (difference between modelled values and ground truth)
    :param max_threshold: Upper boundary for errors
    :return: Array of dates
    """
    df['anomaly'] = False
    # Строки, где расхождение превышает пороговые значения
    df.loc[(df[par] > max_threshold) | (df[par] < min_threshold), 'anomaly'] = True
    anomalous_periods = df[df['anomaly'] == True]['ds'].values

    # График с динамикой показателя и разметкой аномальных дней
    plot_anomalies(df, anomalous_periods)

    return anomalous_periods


# ---------------------- Загрузка исходных данных и визуализация ------------------------

# Все операционные показатели с января 2014 по март 2021 г.
data = load_operational_stats(DATA_LINK)

# Данные по пассажироперевозкам
pass_data = select_parameter(data, 'Passengers carried')
print(pass_data.head())

# График с динамикой пассажироперевозок
plt.plot(pass_data['ds'], pass_data['y'])
plt.title('Пассажироперевозки "Аэрофлота"')
plt.ylabel('тыс. человек')
plt.tight_layout()
plt.show()

# -------------------------------- Прогнозирование -------------------------------------

# Инициализация модели и прогноз
model, forecast = get_forecast(pass_data, N_PERIODS)

# Прогноз и границы доверительного интервала
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(N_PERIODS))

# Визуализация прогноза
fig1 = model.plot(forecast)
a = add_changepoints_to_plot(fig1.gca(), model, forecast)
plt.show()

# Разложение временного ряда на компоненты
fig2 = model.plot_components(forecast)
plt.show()

# -------------------------------- Поиск аномалий ---------------------------------

# 1. За все время, включая исключительно аномальный период с марта 2020 г.

# Расчет отклонений модели от фактических данных
pass_data = calculate_errors(pass_data, forecast, n_periods=N_PERIODS)

# Расхождения в абсолютном выражении (тыс. чел.)
mean_abs_error, abs_error_std = errors_distribution(pass_data, 'error_abs')

# Расхождения в относительном выражении (в долях от 1)
mean_rel_error, abs_rel_std = errors_distribution(pass_data, 'error_rel')

# Аномальные дни за весь период наблюдений
anomalous_days = find_anomalous_days(pass_data, 'error_rel',
                                     min_threshold=mean_rel_error - abs_rel_std,
                                     max_threshold=mean_rel_error + abs_rel_std)

# Аномальные дни за весь период наблюдений
anomalous_days = find_anomalous_days(pass_data, 'error_abs',
                                     min_threshold=mean_abs_error - abs_error_std,
                                     max_threshold=mean_abs_error + abs_error_std)

# 2. За сравнительно стабильный период до 2020 г.

# Берем период до января 2020 г.
pass_data_stable = pass_data[pass_data['ds'] < '2020-02'].copy()

# Повторяем обучение модели, чтобы избежать влияния 2020 г.
# на предшествующие оценки аномальности
model, forecast = get_forecast(pass_data_stable, N_PERIODS)

fig1 = model.plot(forecast)
a = add_changepoints_to_plot(fig1.gca(), model, forecast)
plt.show()

# Расчет отклонений модели от фактических данных
pass_data_stable = calculate_errors(pass_data_stable, forecast, n_periods=N_PERIODS)

# Расхождения в абсолютном выражении (тыс. чел.)
mean_abs_error, abs_error_std = errors_distribution(pass_data_stable, 'error_abs')

# Расхождения в относительном выражении (в долях от 1)
mean_rel_error, abs_rel_std = errors_distribution(pass_data_stable, 'error_rel')

# Аномальные дни за сравнительно стабильный период
anomalous_days = find_anomalous_days(pass_data_stable, 'error_rel',
                                     min_threshold=mean_rel_error - 1.5 * abs_rel_std,
                                     max_threshold=mean_rel_error + 1.5 * abs_rel_std)

# Аномальные дни за сравнительно стабильный период
anomalous_days = find_anomalous_days(pass_data_stable, 'error_abs',
                                     min_threshold=mean_abs_error - 1.5 * abs_error_std,
                                     max_threshold=mean_abs_error + 1.5 * abs_error_std)
