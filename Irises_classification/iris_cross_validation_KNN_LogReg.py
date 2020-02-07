# Анализ 'iris dataset': использование кросс-валидации (cross-validation)
# для выбора модели и ее параметров на основе средней точности.

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

import matplotlib.pyplot as plt

# Считываем данные:
data = load_iris()
X = data.data
y = data.target

# Оцениваем среднюю точность модели KNN при разной величине k:
k_range = range(1, 31)
accuracy = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    accuracy.append(scores.mean())

print('\nСредняя точность модели KNN с величиной k от 1 до 30:')
print(accuracy)

plt.plot(k_range, accuracy)
plt.xlabel('Величина k в KNN')
plt.ylabel('Средняя точность модели')
plt.show()

# Оцениваем среднюю точность модели KNN с k=13:
knn = KNeighborsClassifier(n_neighbors=13)
print('\nСредняя точность модели KNN с k=13:',
      cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())

# Перед созданием модели LogisticRegression обрабатываем исходные данные,
# чтобы привести их к диапазону от 0 до 1:
scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Оцениваем среднюю точность модели LogisticRegression:
lr = LogisticRegression()
print('\nСредняя точность модели LogisticRegression:',
      cross_val_score(lr, X_scaled, y, cv=10, scoring='accuracy').mean())
