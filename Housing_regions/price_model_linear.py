# Линейная модель прогнозирования средней стоимости квадратного метра жилья
# на первичном рынке по регионам и территориям.

import pandas as pd
import numpy as np

import sklearn
from sklearn import linear_model

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Housing_dataset.csv')

# Добавляем столбец с объемом ввода жилья в кв. м на человека:
data['Completions_per_capita'] = data.Housing_completions / data.Population * 1000

# Добавляем столбец с объемом ипотечных кредитов в тыс. руб. на человека:
data['Loans_per_capita'] = data.Loans_vol / data.Population * 1000

# Выводим графики зависимости среднедушевых параметров и цен. Объем ввода жилья
# на душу населения слабо коррелирует с другими рассматриваемыми параметрами.
sns.pairplot(data[['Primary_price', 'Secondary_price',
                   'Income_per_capita', 'Salaries',
                   'Completions_per_capita',
                   'Loans_per_capita', 'Retail_per_capita']],
                    diag_kind="kde", height=1.5)
plt.tight_layout()
plt.show()

# В базе отсутствуют данные о ценах на вторичном рынке в 1 строке
# и данные о ценах на первичном рынке в 5 строках.

print(data.isna().sum())
print(data[data.Secondary_price.isna()])  # Чукотский а/о
print(data[data.Primary_price.isna()])  # Ненецкий а/о, Мурманская область,
# Республика Тыва, Магаданская область, Чукотский а/о

# Для дальнейшего построения модели из таблицы была убрана строка с Чукотским а/о.
# Отсутствующие данные в столбце 'Primary_price' были заполнены на основе типичного
# соотношения цен на первичном и вторичном рынке в других регионах.

# Добавляем столбец с соотношением цен на первичном и вторичном рынке:
data['Primary_to_secondary'] = data.Primary_price / data.Secondary_price

# Вычисляем среднее соотношение цен:
average_dif = data.Primary_to_secondary.mean()

# Заполняем отсутствующие данные в столбце 'Primary_price':
data.Primary_price.fillna(round(data.Secondary_price * average_dif, 2), inplace=True)

# Удаляем строку с отсутствующими данными:
data.dropna(subset=['Primary_price'], inplace=True)

# Отбираем из базы параметры для использования в модели оценки (прогноза)
# среднего уровня цен на первичном рынке жилья:
x = np.array(data[['Income_per_capita', 'Salaries', 'Retail_per_capita']])

# Создаем массив прогнозируемых значений:
y = np.array(data['Primary_price'])

# Каждый из двух массивов разбиваем для две части - для обучения и для тестирования модели.
# Размер выборки для тестирования составит 20% от общего объема базы данных.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# Cоздаем модель, используя линейную регрессию:
linear = linear_model.LinearRegression()

# Передаем модели набор данных для обучения:
linear.fit(x_train, y_train)

# Вычисляем точность прогнозирования с использованием
# нашей модели на основе тестовых данных:
accuracy = linear.score(x_test, y_test)
print('\nTest accuracy: %.2f' % accuracy)

# Выводим коэффициенты линейной зависимости для каждого из параметров,
# определяющих итоговое значение прогнозируемого показателя,
# и сдвиг линии тренда относительно 0:
print('\nLinear coefficients: \n', linear.coef_)
print('\nIntercept: \n', linear.intercept_)

# Тестируем модель на данных, которые не использовались в процессе обучения:
predictions = linear.predict(x_test)

print('\nTesting results:')

# Выводим результат тестирования:
for x in range(len(predictions)):
    print('Expected price:', int(predictions[x]),
          'Actual price:', y_test[x],
          'Accuracy: %.2f' % (predictions[x] / y_test[x] - 1))
