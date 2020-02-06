# Пример классификации объектов 'iris dataset' с использованием модели Logistic Regression.
# Датасет содержит 150 строк с размерами растений по четырем параметрам и названиями видов.

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Загружаем датасет и отделяем входные данные от наименований видов:
X, y = load_iris(return_X_y=True)

# Находим оптимальную величину тестовой выборки:

sizes = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # Рассматриваем вариант от 10% до 40%
accuracy = []  # Создаем список для добавления показателей точности прогноза

for i in sizes:
    # Делим данные на учебную и тестовую части:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i)
    lr = LogisticRegression()  # Создаем модель Logistic Regression
    # Приводим данные к диапазону от 0 до 1 путем деления на максимум:
    divisor = max(X_train.max(), X_test.max())
    X_train = X_train / divisor
    X_test = X_test / divisor
    lr.fit(X_train, y_train)  # Передаем модели учебные данные
    y_pred = lr.predict(X_test)  # Делаем прогноз на массиве тестовых данных
    # Выводим точность прогноза:
    accuracy.append((i, metrics.accuracy_score(y_test, y_pred)))

print('Точность прогноза при различных размерах тестовой выборки:')

for size, acc in accuracy:
    print(f'Размер выборки: {size}. Точность прогноза: {acc}')

# Содаем модель с test_size=0,25:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
lr = LogisticRegression()
divisor = max(X_train.max(), X_test.max())
X_train = X_train / divisor
X_test = X_test / divisor
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Список для расшифровки числовых обозначений видов растений:
iris_types = ['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica']

print('\nПримеры прогноза при тестовой выборке в 25%:')

for i in range(len(y_pred)):
    print(f'Predicted class: {iris_types[y_pred[i]]}. Actual class: {iris_types[y_test[i]]}')
