# Пример классификации объектов 'iris dataset' с использованием модели KNN.
# Датасет содержит 150 строк с размерами растений по четырем параметрам и названиями видов.

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = load_iris()  # Загружаем датасет

X = data.data  # Извлекаем входные данные (размеры)
y = data.target  # Извлекаем итоговые значения (наименования видов)

# Делим данные на учебную и тестовую части в соотношении 60% / 40%:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Находим оптимальную величину K (количество ближайших
# по характеристикам объектов для классификации).

k = range(1, 26)  # Рассматриваем диапазон от 1 до 20

accuracy = [None]  # Создаем список для добавления показателей точности прогноза:
# значению k будет соответствовать индекс элемента, точности прогноза - его значение.

print('Точность прогноза при различных значениях k:')

for i in k:  # В рассматриваемом диапазоне значений k
    knn = KNeighborsClassifier(n_neighbors=i)  # содаем модель KNN,
    knn.fit(X_train, y_train)  # передаем модели учебные данные,
    y_pred = knn.predict(X_test)  # делаем прогноз на массиве тестовых данных,
    accuracy.append(metrics.accuracy_score(y_test, y_pred))  # добавляем в список точность.

# Выводим полученный результат:
for i in range(1, len(accuracy)):
    print(f'k={i} Точность прогноза: {accuracy[i]}')

# Содаем модель с k=10:
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Список для расшифровки числовых обозначений видов растений:
iris_types = ['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica']

print('\nПримеры прогноза при k=10:')

for i in range(len(y_pred)):
    print(f'Predicted class: {iris_types[y_pred[i]]}. Actual class: {iris_types[y_test[i]]}')
