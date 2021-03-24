"""Использование готовой обученной модели из библиотеки transformers
для оценки тональности высказываний (sentiment analysis)
на массиве отзывов о товарах и услугах Amazon, IMDB, YELP.
"""

import pandas as pd
from sklearn import metrics
from transformers import pipeline

from matplotlib import pyplot as plt
import seaborn as sns

# Параметры отображения графиков:
plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams.update({'font.size': 11})

# Считываем данные из csv-файла:
data = pd.read_csv('reviews_data.csv',
                   header=None,
                   names=['review', 'sentiment', 'source'],
                   dtype={'source': 'category'},
                   delimiter='\t')

# Готовая модель для задачи анализа тональности:
nlp = pipeline('sentiment-analysis')

# Пример обработки одного текстового фрагмента:
prediction = nlp('We are very happy to include pipeline into the transformers repository.')
print(prediction)

# Модель требует значительных объемов памяти и много времени на inference:
# обработка массива из 3,000 отзывов может занимать 5-6 минут и до 7-8 Гб.

# Для тестирования модели на меньших объемах исходных данных
# можно взять случайную выборку таким способом:
# data = data.sample(frac=0.1)  # 300 случайных отзывов

# Оценка тональности набора текстов:
input_texts = data['review'].to_list()
prediction = nlp(input_texts)

# Преобразуем список словарей в pd.DataFrame и соединяем с исходными данными:
prediction = pd.DataFrame(prediction)
data['label'] = prediction['label'].values
data['score'] = prediction['score'].values

# Бинарный столбец с прогнозом модели:
data['pred_sentiment'] = data['label'].apply(lambda x: 1 if x == 'POSITIVE' else 0)

# Показатели точности модели:
accuracy = metrics.accuracy_score(data['sentiment'], data['pred_sentiment'])
print('Accuracy score:', accuracy)

# Столбец с вероятностью положительной оценки
data['pred_prob'] = data['score']
data.loc[data['sentiment'] == 0, 'pred_prob'] = 1 - data.loc[data['sentiment'] == 0, 'score']
roc_score = metrics.roc_auc_score(data['sentiment'], data['pred_prob'])
print('ROC AUC:', roc_score)

clf_report = metrics.classification_report(data['sentiment'], data['pred_sentiment'])
print(clf_report)

conf_matrix = metrics.confusion_matrix(data['sentiment'], data['pred_sentiment'])
conf_matrix = pd.DataFrame(conf_matrix,
                           columns=['pred_negative', 'pred_positive'],
                           index=['actual_negative', 'actual_positive'])

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues)
plt.yticks(rotation=0)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('transformer_confusion.png')
plt.show()

# Accuracy score: 0.928
# ROC AUC: 1.0

#               precision    recall  f1-score   support
#
#            0       0.92      0.93      0.93      1500
#            1       0.93      0.92      0.93      1500
#
#     accuracy                           0.93      3000
#    macro avg       0.93      0.93      0.93      3000
# weighted avg       0.93      0.93      0.93      3000
