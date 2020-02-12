import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Считываем данные из csv-файла:
data = pd.read_csv('reviews_data.csv',
                   header=None,
                   names=['review', 'sentiment', 'source'],
                   dtype={'source': 'category'},
                   delimiter='\t')

# Проверяем структуру и типы данных:
print(data.shape)
print(data.dtypes)
print(data)
print(data.groupby('source').sentiment.value_counts())

# Определяем значения X и y отдельно для каждой группы:
X_amazon = data.loc[data['source'] == 'Amazon', 'review']  # Текст отзыва
y_amazon = data.loc[data['source'] == 'Amazon', 'sentiment']  # Категория (0 / 1)

X_imdb = data.loc[data['source'] == 'IMDB', 'review']
y_imdb = data.loc[data['source'] == 'IMDB', 'sentiment']

X_yelp = data.loc[data['source'] == 'YELP', 'review']
y_yelp = data.loc[data['source'] == 'YELP', 'sentiment']

# Делим отзывы с Amazon на учебные и тестовые:
X_train, X_test, y_train, y_test = train_test_split(X_amazon, y_amazon)

# Преобразуем текст отзывов с Amazon в числовую матрицу:
vect = CountVectorizer()
X_train_num = vect.fit_transform(X_train)
X_test_num = vect.transform(X_test)

nb = MultinomialNB()  # Создаем модель
nb.fit(X_train_num, y_train)  # Обучаем модель
y_pred_class = nb.predict(X_test_num)  # Делаем прогноз

print('\nТочность модели на тестовых данных (Amazon):', metrics.accuracy_score(y_test, y_pred_class))

print('\nМатрица ошибок (Amazon):')
print(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred_class),
                   columns=['pred_negative', 'pred_positive'],
                   index=['actual_negative', 'actual_positive']))

print('\nОшибочно классифицированы как позитивные (Amazon):')
print(X_test[y_pred_class > y_test])

print('\nОшибочно классифицированы как негативные (Amazon):')
print(X_test[y_pred_class < y_test])

y_pred_prob = nb.predict_proba(X_test_num)[:, 1]

print('\nПоказатель ROC AUC (Amazon):', metrics.roc_auc_score(y_test, y_pred_prob))

# Делим отзывы с IMDB на учебные и тестовые:
X_train, X_test, y_train, y_test = train_test_split(X_imdb, y_imdb)

# Преобразуем текст отзывов с IMDB в числовую матрицу:
vect = CountVectorizer()
X_train_num = vect.fit_transform(X_train)
X_test_num = vect.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_num, y_train)
y_pred_class = nb.predict(X_test_num)

print('\nТочность модели на тестовых данных (IMDB):', metrics.accuracy_score(y_test, y_pred_class))

print('\nМатрица ошибок (IMDB):')
print(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred_class),
                   columns=['pred_negative', 'pred_positive'],
                   index=['actual_negative', 'actual_positive']))

print('\nОшибочно классифицированы как позитивные (IMDB):')
print(X_test[y_pred_class > y_test])

print('\nОшибочно классифицированы как негативные (IMDB):')
print(X_test[y_pred_class < y_test])

y_pred_prob = nb.predict_proba(X_test_num)[:, 1]

print('\nПоказатель ROC AUC (IMDB):', metrics.roc_auc_score(y_test, y_pred_prob))

# Делим отзывы с Yellow Pages на учебные и тестовые:
X_train, X_test, y_train, y_test = train_test_split(X_yelp, y_yelp)

# Преобразуем текст отзывов с Yellow Pages в числовую матрицу:
vect = CountVectorizer()
X_train_num = vect.fit_transform(X_train)
X_test_num = vect.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_num, y_train)
y_pred_class = nb.predict(X_test_num)

print('\nТочность модели на тестовых данных (Yellow Pages):', metrics.accuracy_score(y_test, y_pred_class))

print('\nМатрица ошибок (Yellow Pages):')
print(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred_class),
                   columns=['pred_negative', 'pred_positive'],
                   index=['actual_negative', 'actual_positive']))

print('\nОшибочно классифицированы как позитивные (Yellow Pages):')
print(X_test[y_pred_class > y_test])

print('\nОшибочно классифицированы как негативные (Yellow Pages):')
print(X_test[y_pred_class < y_test])

y_pred_prob = nb.predict_proba(X_test_num)[:, 1]

print('\nПоказатель ROC AUC (Yellow Pages):', metrics.roc_auc_score(y_test, y_pred_prob))
