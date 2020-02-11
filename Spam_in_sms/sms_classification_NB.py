import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Считываем данные об sms-сообщениях из файла.
# Файл не содержит заголовков столбцов.
# Значения (категория и текст сообщения) разделены знаком табуляции.
sms = pd.read_csv('sms.csv', header=None, names=['category', 'message'], sep='\t')

# Создаем новый столбец с бинарными значениями для категорий сообщений:
sms['category_bin'] = sms.category.map({'ham': 0, 'spam': 1})

# Определяем значения X и y для дальнейшей обработки данных:
X = sms.message   # Текстовые сообщения
y = sms.category_bin  # Категория (0 / 1)

# Делим данные на учебные и тестовые:
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Создаем инструмент векторизации текста:
vect = CountVectorizer()
X_train_num = vect.fit_transform(X_train)  # Преобразуем учебные тексты в числовую матрицу
X_test_num = vect.transform(X_test)  # Преобразуем тестовые тексты в числовую матрицу

# Создаем модель Multinomial Naive Bayes:
nb = MultinomialNB()
nb.fit(X_train_num, y_train)  # Обучаем модель
y_pred_class = nb.predict(X_test_num)  # Делаем прогноз

print('Точность модели:', metrics.accuracy_score(y_test, y_pred_class))

print('\nМатрица ошибок:')
print(pd.DataFrame(metrics.confusion_matrix(y_test, y_pred_class),
                   columns=['pred_ham', 'pred_spam'],
                   index=['actual_ham', 'actual_spam']))

print('\nОшибочно классифицированы как спам:')
print(X_test[y_pred_class > y_test])

print('\nОшибочно классифицированы как не спам:')
print(X_test[y_pred_class < y_test])

y_pred_prob = nb.predict_proba(X_test_num)[:, 1]  # Вероятности отнесения к спаму

print('\nПоказатель ROC AUC:', metrics.roc_auc_score(y_test, y_pred_prob))

tokens_list = vect.get_feature_names()  # Список слов

ham_tokens = nb.feature_count_[0, :]  # Частота слов в сообщениях категории 'ham'
spam_tokens = nb.feature_count_[1, :]  # Частота слов в сообщениях категории 'spam'

# Создаем DataFrame, объединяя данные о частоте слов в двух категориях sms:
tokens = pd.DataFrame({'token': tokens_list, 'ham': ham_tokens,
                       'spam': spam_tokens}).set_index('token')

# Увеличиваем частоту всех слов на 1, чтобы в дальнейшем избежать деления на 0:
tokens['ham'] = tokens.ham + 1
tokens['spam'] = tokens.spam + 1

# Преобразуем абсолютную частоту слов в относительную,
# разделив на частоту соответствующего класса в выборке:
tokens['ham'] = tokens.ham / nb.class_count_[0]
tokens['spam'] = tokens.spam / nb.class_count_[1]

# Добавляем столбец с отношением вероятности встретить слово в спам-сообщениях
# к вероятности встретить это слово в сообщениях из категории 'ham':
tokens['spam_ratio'] = tokens.spam / tokens.ham

print('Слова, наиболее часто встречающиеся в спам-сообщениях:')
print(tokens.sort_values('spam_ratio', ascending=False).head(10))
