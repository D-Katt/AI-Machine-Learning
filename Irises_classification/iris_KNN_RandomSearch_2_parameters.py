# Пример классификации объектов 'iris dataset' с использованием модели KNN.
# Инструмент RandomizedSearchCV используется для настройки 2 гиперпараметров -
# количества "соседей" и весов.

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

data = load_iris()  # Загружаем датасет

X = data.data  # Извлекаем входные данные (размеры)
y = data.target  # Извлекаем итоговые значения (наименования видов)

# Делим данные на учебные и тестовые:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Содаем модель KNN:
knn = KNeighborsClassifier()

# Задаем список значений для настройки параметров:
k_range = range(1, 31)
weight_options = ['uniform', 'distance']
param_dist = {'n_neighbors': k_range, 'weights': weight_options}

# Передаем RandomizedSearchCV оцениваемую модель, список значений параметров,
# указываем разбивку массива на 10 частей во время кросс-валидации,
# 10 комбинаций параметров и random_state для одинаковой разбивки данных:
grid = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=2)

# Передаем учебные данные:
grid.fit(X_train, y_train)

print('Максимальная точность прогноза на тестовой выборке:', grid.best_score_)
print('\nПараметры лучшей модели:', grid.best_estimator_)

y_pred = grid.predict(X_test)  # Делаем прогноз на основе лучшей среди найденных моделей

# Список для расшифровки числовых обозначений видов растений:
iris_types = ['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica']

print('\nПримеры прогноза:')
for i in range(len(y_pred)):
    print(f'\tПрогноз: {iris_types[y_pred[i]]}. Фактическое значение: {iris_types[y_test[i]]}')
