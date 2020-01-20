import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

# Считываем данные из csv-файла (разделитель - ;):
data = pd.read_csv('prices.csv', sep=';')

# Выбираем необходимые для анализа параметры:
data = data[['Room_k', 'House_k', 'Floor_k', 'Condition', 'Sq_m_price']]

# Выбираем параметры, которые небходимо прогнозировать:
predict = 'Sq_m_price'

# Создаем базу, которая содержит все параметры, кроме прогнозируемых:
X = np.array(data.drop([predict], 1))

# Создаем базу, которая содержит только прогнозируемые значения:
y = np.array(data[predict])

# Каждый из двух массивов разбиваем для две части - для обучения и для тестирования модели.
# В данном случае размер выборки для тестирования составит 20% от общего объема базы данных.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

# Для анализа данных мы используем линейную регрессию,
# поэтому создаем соответствующую модель:
linear = linear_model.LinearRegression()

# Передаем модели набор данных для обучения:
linear.fit(x_train, y_train)

print('Training results:')

# Вычисляем и выводим на экран точность прогнозирования с использованием
# нашей модели на основе тестовых данных:
accuracy = linear.score(x_test, y_test)
print('Accuracy: %.2f' % accuracy)

# Выводим коэффициенты линейной зависимости для кадого из параметров,
# определяющих итоговое значение прогнозируемого показателя,
# и сдвиг линии тренда относительно 0:
print('Coefficients: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

# Тестируем модель на данных, которые не использовались в процессе обучения:
predictions = linear.predict(x_test)

print('Testing results:')

# Выводим результат тестирования:
for x in range(len(predictions)):
    print('Expected price:', int(predictions[x]), 'Input data:', x_test[x],
          'Actual price:', y_test[x], 'Accuracy: %.2f' % (predictions[x] / y_test[x] - 1))
