# Пример классификации объектов 'Wine recognition dataset'
# на основе модели SVC и KNN с подбором параметра K через GridSearchCV.

from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

data = load_wine()  # Загружаем данные

x = data.data  # Отделяем параметры
y = data.target  # от наименований классов

# Делим данные на обучающие и тестовые в соотношении 80% / 20%:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

classes = data.target_names  # Список наименований классов

# Создаем модель на основе SVC с линейным типом kernel:
svc = SVC(kernel='linear')

svc.fit(x_train, y_train)  # Обучаем модель
y_pred = svc.predict(x_test)  # Делаем прогноз

acc = metrics.accuracy_score(y_test, y_pred)
print('Точность модели SVC:', acc)

# Создаем модель на основе KNN:
knn = KNeighborsClassifier()

# Выбираем оптимальную величину K из диапазона от 2 до 25:
k_range = range(2, 26)
param_grid = {'n_neighbors': k_range}
grid = GridSearchCV(knn, param_grid, scoring='accuracy')

grid.fit(x_train, y_train)  # Обучаем модель
y2_pred = grid.predict(x_test)  # Делаем прогноз

acc2 = metrics.accuracy_score(y_test, y2_pred)
print('Точность модели KNN:', acc2)

print('\nМаксимальная точность прогноза KNN на тестовой выборке:', grid.best_score_)
print('\nПараметры лучшей модели KNN:', grid.best_estimator_)
