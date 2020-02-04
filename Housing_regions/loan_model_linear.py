# Линейная модель прогнозирования объемов ипотечного кредитования по регионам и территориям.

import pandas as pd
import numpy as np

import sklearn
from sklearn import linear_model

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Housing_dataset.csv')

# Добавляем столбец с совокупным доходом населения в млн. рублей:
data['Total_income'] = data.Population * data.Income_per_capita / 1000000

# Добавляем столбец с бинарной характеристикой миграции: 0 - убыль населения,
# 1 - миграционный прирост населения.
data['Migration_bin'] = 0
data.loc[data.Migration > 0, 'Migration_bin'] = 1

# Выводим графики зависимости параметров. Общий объем кредитования коррелирует
# с численностью населения, общим объемом ввода жилья, совокупным доходом
# и оборотом розничной торговли.
sns.pairplot(data[['Loans_vol', 'Population', 'Housing_completions',
                   'Retail', 'Total_income', 'Salaries', 'Migration']],
                    diag_kind="kde", height=1.5)
plt.tight_layout()
plt.show()

# Отбираем из базы параметры для использования в модели оценки (прогноза)
# объемов ипотечного кредитования и преобразуем в массив:
x = np.array(data[['Population', 'Housing_completions', 'Total_income', 'Retail']])

# Создаем массив прогнозируемых значений:
y = np.array(data['Loans_vol'])

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

# Делаем прогнозы на основе всей выборки тестовых данных:
predictions = linear.predict(x_test)

print('\nTesting results:')

# Выводим сравнение прогнозируемых и фактических показателей по тестовой выборке:
for x in range(len(predictions)):
    print('Expected volume:', int(predictions[x]),
          'Actual volume:', y_test[x],
          'Accuracy: %.2f' % (predictions[x] / y_test[x] - 1))
