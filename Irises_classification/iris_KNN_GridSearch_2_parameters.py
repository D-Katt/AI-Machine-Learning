# Пример классификации объектов 'iris dataset' с использованием модели KNN.
# Инструмент GridSearchCV используется для настройки 2 гиперпараметров -
# количества "соседей" и весов.

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

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
param_grid = {'n_neighbors': k_range, 'weights': weight_options}

# Передаем GridSearchCV оцениваемую модель, список значений параметров
# для выбора лучшего и критерий оценки, указываем разбивку массива
# на 10 частей во время кросс-валидации:
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')

# Передаем учебные данные:
grid.fit(X_train, y_train)

print('Максимальная точность прогноза на тестовой выборке:', grid.best_score_)
print('\nПараметры лучшей модели:', grid.best_estimator_)

# Просмотреть среднюю точность и стандартное отклонение по всем вариантам:
results = pd.DataFrame(grid.cv_results_)
print(results[['param_n_neighbors', 'param_weights', 'mean_test_score', 'std_test_score']])
