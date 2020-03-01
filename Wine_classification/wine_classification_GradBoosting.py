# Пример классификации объектов 'Wine recognition dataset' на основе модели GradientBoostingClassifier.

import pandas as pd

from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = load_wine()  # Загружаем данные

X = data.data  # Отделяем параметры
y = data.target  # от категорий напитков

parameters = data.feature_names  # Список наименований параметров
target_names = data.target_names  # Список наименований классов

# Приводим значения параметров к диапазону от 0 до 1:
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Делим данные на учебную и тестовую части:
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# Создаем модель:
clf = GradientBoostingClassifier(n_estimators=100,
                                 learning_rate=1.0,
                                 max_depth=1,
                                 random_state=0).fit(X_train, y_train)

y_pred = clf.predict(X_test)  # Делаем прогноз для тестовых данных

# Оцениваем точность, сравнивая прогноз с фактическими значениями:
print('Accuracy score for test data:', metrics.accuracy_score(y_test, y_pred))

# То же самое в одну строку:
print('\nMean accuracy for test data:', clf.score(X_test, y_test))

# Рассчитываем вероятности отнесения объектов к разным классам:
y_pred_prob = clf.predict_proba(X_test)
y_test_classes = pd.get_dummies(y_test)

# Выводим показатель ROC AUC:
print('\nROC AUC:', metrics.roc_auc_score(y_test_classes, y_pred_prob))

# Создадим confusion_matrix для анализа результатов и посмотрим,
# в классификации объектов какого класса модель чаще делает ошибки:
col_names = ['pred_' + i for i in target_names]
ind = ['fact_' + i for i in target_names]

conf_matrix = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred),
                           columns=col_names,
                           index=ind)

print('\nConfusion matrix:\n', conf_matrix)

# Посмотрим на чувствительность модели к определению классов:
rec_score = metrics.recall_score(y_test, y_pred, average=None)

print('\nЧувствительность модели к определению классов:')
for i in range(len(target_names)):
    print(f'Класс {target_names[i]}: {rec_score[i]}')
