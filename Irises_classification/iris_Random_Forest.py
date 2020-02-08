# Пример классификации объектов 'iris dataset' с использованием модели Random Forest.
# Датасет содержит 150 строк с размерами растений по четырем параметрам и названиями видов.

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = load_iris()  # Загружаем датасет

X = data.data  # Извлекаем входные данные (размеры)
y = data.target  # Извлекаем итоговые значения (наименования видов)

# Делим данные на учебные и тестовые:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Содаем модель Random Forest на 50 деревье:
rf = RandomForestClassifier(n_estimators=50)
rf.fit(X_train, y_train)

print('Средняя точность прогноза на тестовой выборке:', rf.score(X_test, y_test))

y_pred = rf.predict(X_test)  # Прогноз на тестовой выборке

# Список для расшифровки числовых обозначений видов растений:
iris_types = ['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica']

print('\nПримеры прогноза:')

for i in range(len(y_pred)):
    print(f'\tПрогноз: {iris_types[y_pred[i]]}. Фактическое значение: {iris_types[y_test[i]]}')
