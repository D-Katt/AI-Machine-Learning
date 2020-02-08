# Пример классификации объектов 'iris dataset' с использованием модели Random Forest.
# Инструмент RandomizedSearchCV используется для настройки 2 гиперпараметров -
# кол-ва деревьев и весов.

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

data = load_iris()  # Загружаем датасет

X = data.data  # Извлекаем входные данные (размеры)
y = data.target  # Извлекаем итоговые значения (наименования видов)

# Делим данные на учебные и тестовые:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Содаем модель Random Forest:
rf = RandomForestClassifier()

# Задаем список значений для настройки параметров:
n_estimators = range(10, 70)
criterion = ['gini', 'entropy']
param_dist = {'n_estimators': n_estimators, 'criterion': criterion}

# Передаем RandomizedSearchCV оцениваемую модель, список значений параметров,
# указываем разбивку массива на 10 частей во время кросс-валидации,
# 10 комбинаций параметров и random_state для одинаковой разбивки данных:
grid = RandomizedSearchCV(rf, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=2)

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
