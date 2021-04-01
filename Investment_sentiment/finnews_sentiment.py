"""Сравнение нескольких моделей машинного обучения для задачи классификации
тональности новостных сообщений. Модели могут использользоваться
для прогнозирования поведения частных инвесторов."""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

# Параметры отображения таблиц
pd.set_option('display.max_colwidth', 250)

# Параметры отображения графиков
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams.update({'font.size': 11})


# ----------------------- Анализ данных и визуализация -------------------------

data = pd.read_csv('all-data.csv',
                   header=None,
                   names=['sentiment', 'text'],
                   encoding='latin-1',
                   dtype={'sentiment': 'category'})

# Объем данных небольшой и содержит около 5,000 фрагментов новостных сообщений 3 классов:
# негативные, нейтральные и позитивные новости.
# Тексты различаются по длине и стилю, могут содержать слова, числа и проценты,
# неправильную пунктуацию, двойные пробелы и проч.
data.head()

# Распределение текстов по классам: наиболее широко представлены нейтральные новости,
# негативные новости составляют только 12.5% от общего числа текстов.
classes_distribution = data['sentiment'].value_counts(normalize=True)
print(classes_distribution)

labels = classes_distribution.index
values = classes_distribution.values
plt.pie(values, labels=labels, startangle=90, autopct='%1.1f%%')
plt.title('Distribution of Classes')
plt.tight_layout()
plt.show()

# Длина текстов (количество слов)
data['n_words'] = data['text'].str.split()
data['n_words'] = data['n_words'].apply(lambda x: [word for word in x if len(word) > 1])  # удаляем слова до 2 символов
data['n_words'] = data['n_words'].apply(lambda x: len(x))

min_length = data['n_words'].min()
max_length = data['n_words'].max()
mean_length = data['n_words'].mean()
median_length = data['n_words'].median()

plt.hist(data['n_words'], bins=10)
plt.axvline(mean_length, color='red', label='Mean')
plt.axvline(median_length, color='green', label='Median')
plt.legend()
plt.title('Headlines Length')
plt.xlabel('Number of words')
plt.ylabel('Frequency')
plt.show()

# Длина текстов варьирует от 1 до 50 слов (после удаления слов из одного символа).
# Медианная длина текста - около 20 слов.

print(f'Sentence length: {min_length} - {max_length} words\n'
      f'Mean length = {mean_length}\nMedian length = {median_length}')

# Примеры коротких текстов
print(data[data['n_words'] < 5])

# ---------------------------- Функции и создание моделей ---------------------------


def classification_heatmap(cm):
    """Функция отображает график к матрицей ошибок классификации."""
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Reds)
    plt.yticks(rotation=0)
    plt.title('Confusion Matrix')
    plt.show()


def accuracy_estimator(model, name):
    """Функция оценивает показатели точности модели."""
    # Прогноз на тестовых данных
    y_pred_class = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_class)
    print(f'{name} model accuracy: {acc}')
    
    conf_matrix = confusion_matrix(y_test, y_pred_class)
    classes_names = le.classes_
    columns = ['pred_' + name for name in classes_names]
    indexes = ['actual_' + name for name in classes_names]
    conf_matrix = pd.DataFrame(conf_matrix, columns=columns, index=indexes)
    classification_heatmap(conf_matrix)
    
    cls_report = classification_report(y_test, y_pred_class, target_names=classes_names)
    print(cls_report)
    
    try:
        y_pred_prob = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
        print(f'ROC AUC = {roc_auc}')
    except Exception as e:
        print('Probability estimations and ROC AUC are not available.')


def sentiment_reader(model, name):
    """Функция создает pipeline, включающий предварительную обработку данных
    и модель классификации, обучает модель на учебной выборке и
    вызывает функцию оценки точности модели на тестовой выборке."""
    pipe = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 3), max_df=0.8)),  # Преобразуем слова в токены,
        # используя отдельные слова, пары и тройки слов.
        ('tfidf', TfidfTransformer()),  # Принимаем во внимание частоту употребления слов
        ('clf', model)
    ])
    pipe.fit(X_train, y_train)
    accuracy_estimator(pipe, name)


# ----------------------------- Сравниваем базовые модели sklearn ----------------------------

# Кодируем названия категорий
le = LabelEncoder()
y = le.fit_transform(data['sentiment'])
print(le.classes_)

# Оставляет 20% исходных данных для тестирования.
# Принимаем во внимание дисбаланс классов.
X_train, X_test, y_train, y_test = train_test_split(data['text'],
                                                    y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=1)

# Оцениваем модель NaiveBayes с базовыми параметрами
sentiment_reader(MultinomialNB(), 'NaiveBayes')

# Оцениваем модель LogisticRegression с использованием параметра class_weight='balanced'.
# Задаем большое значение параметра 'max_iter', чтобы модель сошлась.
sentiment_reader(LogisticRegression(class_weight='balanced', max_iter=1000), 'LogisticRegression')

# Оцениваем модель SGDClassifier с параметрами linear SVM class_weight='balanced'.
sentiment_reader(SGDClassifier(class_weight='balanced', loss='hinge', penalty='l2', tol=None), 'SGDClassifier')

# Модель SGDClassifier показала наиболее высокую точность среди 3 протестированных моделей,
# но ошибка в определении наименее распространенных классов осталась значительной.
# Эксперименты с оптимизацией параметров модели через GridSearch существенно не улучшили ситуацию.

# ------------------------------ Балансирование классов ---------------------------------

# Варианты балансировки классов:
# - представить все 3 класса в учебной и тестовой выборке в равной пропорции
# - Уменьшить количество примеров наиболее распространного класса,
#   чтобы уравнять его со вторым по распространности.

# Тестирование показало, что первый подход дает лучшие результаты.

# Количество текстов в наименее распространном классе
quota = data['sentiment'].value_counts().min()
print(quota)

# Новый датафрейм для добавления равного количество экземпляров каждого класса
balanced_data = pd.DataFrame(columns=['sentiment', 'text'])

data_groups = data.groupby('sentiment')
for group in data_groups.indices:
    reduced_class = data_groups.get_group(group)[['sentiment', 'text']].iloc[:quota, :]
    balanced_data = balanced_data.append(reduced_class, ignore_index=True)

# Распределение классов в новом датасете
print(balanced_data['sentiment'].value_counts())

# Прогнозируемые значения
y = le.fit_transform(balanced_data['sentiment'])

# Учебная и тестовая выборки
X_train, X_test, y_train, y_test = train_test_split(balanced_data['text'],
                                                    y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=1)

# Оцениваем SGDClassifier model
sentiment_reader(SGDClassifier(loss='hinge', penalty='l2', tol=None), 'SGDClassifier')

# Все показатели существенно улучшились
# - Recall для негативных и положительных новостей вырос. Accuracy для нейтральных новостей
#   немного снизился по сравнению с вариантом, когда модель обучается на всем массиве данных.
# - Precision увеличился для наименее представленных классов без негативных последствий для нейтрального класса.

# Проверяем, как модель поведет себя, если обучить ее на данных с классами в равной пропорции
# без выделения тестовой выборки, а потом оценить точность на полном массиве несбалансированных по классам текстов.

X = balanced_data['text']

pipe = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 3), max_df=0.8)),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2', tol=None))
    ])
pipe.fit(X, y)

print(pipe.score(X, y))

# Прогноз для всего исходного массива текстов
unreduced_X = data['text']
unreduced_y = le.transform(data['sentiment'])
print(pipe.score(unreduced_X, unreduced_y))

# Матрица ошибок
predicted_y = pipe.predict(unreduced_X)
conf_matrix = confusion_matrix(unreduced_y, predicted_y)

classes_names = le.classes_
columns = ['pred_' + name for name in classes_names]
indexes = ['actual_' + name for name in classes_names]
conf_matrix = pd.DataFrame(conf_matrix, columns=columns, index=indexes)

classification_heatmap(conf_matrix)

# По сравнению с моделью, которая обучалась на несбалансированных данных,
# точность прогноза улучшилась. Модель научилась отличать негативные новости от других классов.
# Однако следует помнить, что в этом случае модель видела все экземпляры негативных новостей.
# Будет ли точность такой же высокой при тестировании на новых данных, не очевидно.
# Сохранились ошибки между классами положительных и нейтральных новостей,
# что свидетельствует о том, что используемый подход только частично улучшил
# прежние результаты.

# Оптимальным решением этой проблемы несбалансированных классов было бы расширение
# датасета с добавлением новых примеров негативных и положительных новостей,
# с тем чтобы все три класса были представлены примерно в равной пропорции.

# Пробуем получить оценки нового текста:
new_samples = ['Experts expect the world economy to grow at a steady rate of 3% a year.',
               'Local retailers reported much lower revenues this year. Expansion plans are suspended.']
prediction = pipe.predict(new_samples)
for pred in prediction:
    print(le.classes_[pred])
