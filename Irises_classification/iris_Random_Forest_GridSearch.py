# Пример классификации объектов 'iris dataset' с использованием модели Random Forest.
# Инструмент GridSearchCV используется для настройки гиперпараметра - кол-ва деревье.

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

data = load_iris()  # Загружаем датасет

X = data.data  # Извлекаем входные данные (размеры)
y = data.target  # Извлекаем итоговые значения (наименования видов)

# Делим данные на учебные и тестовые:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Содаем модель Random Forest:
rf = RandomForestClassifier()

# Задаем список значений для настройки параметра "количество деревьев":
param_grid = {'n_estimators': [20, 30, 40, 50, 60, 70]}

# Передаем GridSearchCV оцениваемую модель, список значений параметра
# для выбора лучшего и критерий оценки:
grid = GridSearchCV(rf, param_grid, scoring='accuracy')

# Передаем учебные данные:
grid.fit(X_train, y_train)

print('Максимальная точность прогноза на тестовой выборке:', grid.best_score_)
print('\nПараметры лучшей модели:', grid.best_estimator_)

y_pred = grid.predict(X_test)  # Прогноз на тестовой выборке
# 'grid' хранит параметры лучшей модели.

# Список для расшифровки числовых обозначений видов растений:
iris_types = ['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica']

print('\nПримеры прогноза:')

for i in range(len(y_pred)):
    print(f'\tПрогноз: {iris_types[y_pred[i]]}. Фактическое значение: {iris_types[y_test[i]]}')
