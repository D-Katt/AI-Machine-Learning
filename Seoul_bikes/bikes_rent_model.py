"""Прогнозирование спроса на аренду велосипедов.
"""

import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor

import seaborn as sns
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Параметры отображения датафреймов:
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 25)

# Параметры отображения графиков:
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams.update({'font.size': 11})

# ---------------------- Загрузка и обработка исходных данных ------------------------------

# Ссылка на csv-файл с данными:
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv'

# Исходный файл в заголовках столбцов таблицы содержит символы, которые некорректно считываются при парсинге.
# Поэтому считываем только данные и добавляем новые заголовки столбцов.
# Одновременно преобразуем часть столбцов в категорийные, столбец с датами - в формат datetime.

# Новый список заголовков:
cols = ['date', 'n_bikes', 'hour', 'temperature', 'humidity', 'windspeed', 'visibility',
        'dew_point_temp', 'solar_radiation', 'rainfall', 'snowfall', 'season', 'holiday', 'func_day']

# Категорийные столбцы:
dtypes = {'season': 'category', 'holiday': 'category', 'func_day': 'category'}

# Считываем данные:
data = pd.read_csv(url, skiprows=1, header=None, names=cols, parse_dates=['date'], dtype=dtypes)

# Данные:
print(data.shape)
print(data.dtypes)
print(data.head())

# Проверяем наличие пропусков в данных:
data.isna().sum()

# Добавляем столбец с номером года:
data['year'] = data['date'].dt.year

# Добавляем столбец с номером месяца:
data['month'] = data['date'].dt.month

# Сводные ежедневные данные:
daily_data = data.groupby('date')['n_bikes'].sum().reset_index()

# -------------------------------- Визуализация данных ---------------------------------------

plt.plot(daily_data['date'], daily_data['n_bikes'])
plt.ylabel('Кол-во велосипедов, шт. в сутки')
plt.title('Динамика спроса на аренду велосипедов')
plt.show()

# Из графика с динамикой числа аредованных велосипедов видно, что период до декабря 2017 года
# принципиально отличается по тренду и амплитуде колебаний рассматриваемого показателя от последующего периода.

data.hist(bins=20, figsize=(14, 10))
plt.suptitle('Распределение значений параметров')
plt.show()

# Анализ гистограмм распределения параметров показывает, что количество данных за 2017 год
# в несколько раз меньше, чем за 2018 год. То есть до декабря 2017 года данные представлены не за все периоды.

# Количество данных, приходящихся на 2017 и 2018 год:
data['year'].value_counts()

# Для дальнейшего анализа берем данные с декабря 2017 года:
data = data[data['date'] > '2017-12']

# Средние данные по времени суток:
hourly_average = data.groupby('hour')['n_bikes'].mean().reset_index()

plt.bar(hourly_average['hour'], hourly_average['n_bikes'])
plt.xlabel('Время суток')
plt.ylabel('Кол-во велосипедов, шт./ч')
plt.title('Изменение спроса с течение дня')
plt.show()

# В течение суток прослеживается два явно выраженных пика спроса на аренду велосипедов:
# наиболее значительный - в вечерние часы (18.00 - в среднем около 1500 шт. в час)
# и менее значительный - в утренние (8.00 - чуть более 1000 шт. в час).

# Средние данные по месяцам:
monthly_average = data.groupby('month')['n_bikes'].mean().reset_index()

plt.bar(monthly_average['month'], monthly_average['n_bikes'])
plt.xlabel('Месяц')
plt.ylabel('Кол-во велосипедов, шт./ч')
plt.title('Изменение спроса в течение года')
plt.show()

# В течение года пик спроса на велосипеды приходится на июнь (в среднем около 1000 шт. в час).
# В октябре также происходит увеличение спроса (в среднем до 850 шт. в час).
# Наименьший спрос приходится на зимние месяцы (в среднем около 400 шт. в час).

# Начало временного ряда:
print(data['date'].min())

# Окончание временного ряда:
print(data['date'].max())

# Распределение спроса по сезонам:
seasonal_data = data.groupby('season')['n_bikes'].sum().reset_index()

plt.pie(seasonal_data['n_bikes'], labels=seasonal_data['season'], autopct='%1.1f%%')
plt.title('Структура спроса по сезонам')
plt.show()

# Наиболее значительный объем спроса приходится на летний период (37.5%),
# наименее значительный - на зимний (6.7%).
# В осенний период спрос чуть более активен, чем в весенний.

correlation = data.corr()
ax = sns.heatmap(correlation, center=0, annot=True, cmap='RdBu_r')
l, r = ax.get_ylim()
ax.set_ylim(l + 0.5, r - 0.5)
plt.yticks(rotation=0)
plt.title('Матрица корреляции')
plt.show()

# Параметры с наиболее высокой корреляцией со спросом на аренду велосипедов:
# - Температура воздуха
# - Час суток
# - Температура "точки росы"
# - Солнечная радиация

# --------------------------- Подготовка данных для модели -------------------------------

# Прогнозируемые значения:
y = data.pop('n_bikes')

# Входные параметры для моделей:
X = data.select_dtypes(include=['number', 'category'])

# Составляем списки числовых и категорийных параметров:
num_cols = data.select_dtypes(include='number').columns
cat_cols = data.select_dtypes(include='category').columns

# Преобразование данных (кодирование категорийных столбцов,
# нормирование числовых значений):
ct = make_column_transformer(
    (OneHotEncoder(), cat_cols),
    (StandardScaler(), num_cols),
    remainder='passthrough'
)

# Нормирование данных:
X_processed = ct.fit_transform(X)

# ------------------------------------ Подбор модели ------------------------------------

# Инструмент для разбиения данных на группы при кросс-валидации:
kf = KFold(10, shuffle=True)

# Список рассматриваемых моделей:
models = [('KNN', KNeighborsRegressor()),
          ('Decision Tree', DecisionTreeRegressor()),
          ('Gradient Boosting', GradientBoostingRegressor()),
          ('Random Forest', RandomForestRegressor(n_estimators=100))]

# Параметр R2 при кросс-валидации для рассматриваемых моделей:
for name, model in models:
    rmodel = model
    r2 = cross_val_score(rmodel, X_processed, y, cv=kf, scoring='r2').mean()
    print(f'{name}: R2 = {r2}')

# KNN: R2 = 0.7906359080775422
# Decision Tree: R2 = 0.7634230851528636
# Gradient Boosting: R2 = 0.8449562859736066
# Random Forest: R2 = 0.878704138513225

# Регрессоры с наиболее высокой точностью:
# - Random Forest
# - Gradient Boosting

# Создадим ансамбль из двух алгоритмов регрессии:
r_1 = RandomForestRegressor(n_estimators=100)
r_2 = GradientBoostingRegressor()
model = VotingRegressor([('RF', r_1), ('GB', r_2)])

# Точночть прогнозов при кросс-валидации:
print(cross_val_score(model, X_processed, y, cv=kf, scoring='r2').mean())
# 0.8733412114143849

# Полученный результат комбинированной модели сопоставим результатом модели Random Forest
# и не улучшил точность прогнозов.
# Для улучшения модели целесообразно взять для обучения выборку данных
# за более продолжительный период времени (за несколько лет),
# чтобы сезонные колебания спроса и общий тренд учитывались более корректно.
# Однако в имеющемся датасете около половины от всех доступных данных
# не могут использоваться для анализа, т.к. принципиально отличаются по динамике от последующих значений.

# При отсутствии дополнительных данных пробуем улучшить
# точность модели подбором оптимальных гиперпараметров:
model = RandomForestRegressor()
params = {'n_estimators': [50, 70, 100, 120],
          'min_samples_split': [2, 3]}
grid = GridSearchCV(model, params, cv=kf, scoring='r2', n_jobs=-1, refit=True)
grid.fit(X_processed, y)

# Параметры оптимальной модели:
print(grid.best_estimator_)

# Точность оптимальной модели:
print(grid.best_score_)
# 0.8803249330116183

# По итогам кросс-валидации выбраны те же параметры модели RandomForestRegressor, которые заданы по умолчанию.
# Полученное незначительное увеличение точности при кросс-валидации может быть связано
# с особенностями разбиения данных на группы (параметр random_state не был задан).

# С учетом качества и объема данных, используемых для обучения, созданная прогнозная модель
# может использоваться только для оценки спроса на аренду велосипедов в краткосрочной перспективе.
# Помимо сезонного фактора и времени суток, значимо влияющих на спрос,
# в качестве входных параметров используются детальные погодно-климатические условия,
# прогноз которых сам по себе имеет высокую степень погрешности.
# Поэтому наибольшую достоверность будут иметь прогнозы, составляемые не более чем на 2-3 дня вперед.
# С увеличением горизонта прогнозирования до 10 дней и более точность оценок будет резко снижаться.
