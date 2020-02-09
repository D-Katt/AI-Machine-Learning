# Пример классификации объектов 'Wine recognition dataset' на основе модели LogisticRegression.
# Сравнение показателей 'accuracy' и 'null accuracy' для модели с ограниченным числом параметров.
# Применение 'confusion_matrix' для анализа ошибок в классификации и расчет показателя 'recall_score'.
# Расчет показателя 'ROC AUC' для моделей с одним параметром и со всеми параметрами.

import pandas as pd
import numpy as np

from sklearn.datasets import load_wine
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = load_wine()  # Загружаем данные

X = data.data  # Отделяем параметры
y = data.target  # от категорий напитков

parameters = data.feature_names  # Список наименований параметров
target_names = data.target_names  # Список наименований классов

# Предварительно обрабатываем значения параметров
# и приводим их к диапазону от 0 до 1:
scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Преобразуем данные в pandas DataFrame:
wine = pd.DataFrame(X_scaled)
wine.columns = parameters
wine['class'] = y

print('Корреляция класса с другими параметрами:\n', wine.corr()['class'])

print('\nРаспределение данных по классам:\n', wine['class'].value_counts(normalize=True))

# Для примера построим модель, которая принимает в расчет только параметр 'nonflavanoid_phenols':
X_train, X_test, y_train, y_test = train_test_split(np.array(wine.nonflavanoid_phenols)
                                                    .reshape(-1, 1), y, random_state=2)

lr = LogisticRegression()  # Создаем модель
lr.fit(X_train, y_train)  # Обучаем модель

y_pred = lr.predict(X_test)  # Делаем прогноз
print('\nТочность модели с 1 параметром:', metrics.accuracy_score(y_test, y_pred))

# Точность модели с 1 параметром несущественно превышает долю
# наиболее широко представленного в выборке класса объектов.
# То есть модель, которая всегда выбирает самый распространенный класс,
# демонстрировала бы сопоставимую эффективность.

# Создадим confusion_matrix для анализа результатов и посмотрим,
# в классификации объектов какого класса модель чаще делает ошибки:
col_names = ['pred_' + i for i in target_names]
ind = ['fact_' + i for i in target_names]

conf_matrix = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred),
                           columns=col_names,
                           index=ind)

print('\nМатрица ошибок модели с 1 параметром:\n', conf_matrix)

rec_score = metrics.recall_score(y_test, y_pred, average=None)

print('\nЧувствительность модели с 1 параметром к определению классов):')
for i in range(len(target_names)):
    print(f'Класс {target_names[i]}: {rec_score[i]}')

y_pred_prob = lr.predict_proba(X_test)  # Вероятности отнесения тестовых объектов к 3 классам.
y_test_classes = pd.get_dummies(y_test)  # Фактический класс в виде матрицы с 3 столбцами,
# соответствующими 3 классам и содержащими значения 0 и 1.

print('\nROC AUC модели с 1 параметром:', metrics.roc_auc_score(y_test_classes, y_pred_prob))

# Построим модель для анализа всех доступных параметров:
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=2)

lr = LogisticRegression()  # Создаем модель
lr.fit(X_train, y_train)  # Обучаем модель

y_pred = lr.predict(X_test)  # Делаем прогноз
print('\nТочность модели со всеми параметром:', metrics.accuracy_score(y_test, y_pred))

y_pred_prob = lr.predict_proba(X_test)
y_test_classes = pd.get_dummies(y_test)

print('\nROC AUC модели со всеми параметрами:', metrics.roc_auc_score(y_test_classes, y_pred_prob))
