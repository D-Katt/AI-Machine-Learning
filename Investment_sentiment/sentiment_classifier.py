"""Пример transfer learning: настройка готовой модели с TF Hub
для классификации тональности финансовых новостей с точки зрения частных инвесторов.
К готовой модели добавляются новые слои для маркирования текстов по 3 классам:
негативные, нейтральные и позитивные новости.
Исходная модель обучалась на основе датасета English Google News (130 Гб).
"""

import tensorflow_hub as hub
import tensorflow as tf

import pandas as pd
import numpy as np
import string

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# Параметры отображения таблиц
pd.set_option('display.max_colwidth', 250)

# Параметры отображения графиков
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = 9, 6
plt.rcParams.update({'font.size': 11})

# Переменная определяет, будут ли классы балансироваться перед обучением
BALANCED = True

# Исходные данные: 4,846 образцов текста с размеченными категориями
data = pd.read_csv('all_data.csv',
                   header=None,
                   names=['sentiment', 'text'],
                   encoding='latin-1',
                   dtype={'sentiment': 'category'})

# ---------------------- EDA и обработка текста ------------------------

# Соотношение классов
classes_distribution = data['sentiment'].value_counts()

labels = classes_distribution.index
values = classes_distribution.values
plt.pie(values, labels=labels, startangle=90, autopct='%1.1f%%')
plt.title('Distribution of Classes')
plt.tight_layout()
plt.show()

# Готовая модель принимает на вход пакет текстов как одномерный тенсор
# и обрабатывает его, разбивая текст по пробелам.
# Удаление пунктуации и другие операции по "очистке" текста в модели не производятся.


def text_cleaning(s: str):
    """Функция удаляет знаки препинания и двойные пробелы,
    слова, содержащие менее 2 символов, преобразует текст
    в нижний регистр."""
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = s.replace('  ', ' ').lower()
    s = ' '.join([word for word in s.split() if (len(word) > 1 and word.isalpha())])
    return s


data['text_cleaned'] = data['text'].apply(text_cleaning)

# Длина текстов
data['n_words'] = data['text_cleaned'].apply(lambda x: len(x.split()))

min_length = data['n_words'].min()
max_length = data['n_words'].max()
mean_length = data['n_words'].mean()
median_length = data['n_words'].median()

print(f'Sentence length: {min_length} - {max_length} words\nMean length = '
      f'{mean_length}\nMedian length = {median_length}')

plt.hist(data['n_words'], bins=10)
plt.axvline(mean_length, color='red', label='Mean')
plt.axvline(median_length, color='green', label='Median')
plt.legend()
plt.title('Headlines Length')
plt.xlabel('Number of words')
plt.ylabel('Frequency')
plt.show()

if BALANCED:
    # Количество текстов в наиболее распространном классе
    quota = data['sentiment'].value_counts().min()
    # Новый DataFrame для добавления одинакового количества текстов из каждого класса
    balanced_data = pd.DataFrame(columns=['sentiment', 'text_cleaned'])
    # Уменьшаем количество текстов в каждом классе до выбранного лимита
    data_groups = data.groupby('sentiment')
    for group in data_groups.indices:
        reduced_class = data_groups.get_group(group)[['sentiment', 'text_cleaned']].iloc[:quota, :]
        balanced_data = balanced_data.append(reduced_class, ignore_index=True)
    # Обновляем исходный DataFrame
    data = balanced_data
    del balanced_data

# Входные данные для модели в качестве предварительно очищенных фрагментов текста
X = data['text_cleaned']

# Категории
class_names = ['negative', 'neutral', 'positive']

# Бинарные столбцы для каждой категории
for category in class_names:
    data[category] = data['sentiment'].apply(lambda x: 1 if x == category else 0)

# Прогнозируемые значения для модели
y = data[class_names].values

# Оставляем 20% исходных данных для тестирования.
# Принимаем во внимание дисбаланс классов при разбиении на учебную и тестовую выборки.
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=1)

# Преобразуем данные в тенсоры
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)

# ------------------------ Загрузка и модификация модели --------------------------

# При обработке текста модели использует "Swivel co-occurrence matrix" и встроенные символы OOV
# (Out of Vocabulary). Текст преобразуется в 20-мерный вектор.
# Словарь содержит 20,000 токенов и 1 символ для не известных слов.

# Загрузка модели с TF Hub
hub_layer = hub.KerasLayer('https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1',
                           output_shape=[20],
                           input_shape=[],
                           dtype=tf.string,
                           trainable=True)

# Sequential model с дополнительными слоями.
# Для избежания переобучения на маленьком датасете добавлена регуляризация.
model = tf.keras.Sequential(
    [
        hub_layer,  # Original model
        tf.keras.layers.Dense(64, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # Fully connected layer
        tf.keras.layers.Dense(32, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(3, activation='softmax')  # Final layer for 3 classes
    ]
)

model.summary()

# Метрики и оптимизатор
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

EPOCHS = 500

# Обучение будет остановлено при отсутствии прогресса в течение 10 эпох
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                              patience=10,
                                              restore_best_weights=True)

# Словарь весов классов
if BALANCED:
    balancing_dict = {index: 1 / 3 for index in range(3)}
# Если используются исходные данные с дисбалансом классов,
# придаем больший вес наименее представленным категориям.
else:
    balancing_dict = {0: 0.50, 1: 0.15, 2: 0.35}

# Обучение модели (20% данных используются для валидации)
history = model.fit(X_train, y_train,
                    batch_size=32, validation_batch_size=32,
                    validation_split=0.2, shuffle=True,
                    class_weight=balancing_dict,
                    callbacks=[early_stop],
                    epochs=EPOCHS, verbose=2)

# Визуализация метрик в процессе обучения и валидации
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training')
plt.plot(epochs, val_loss, 'ro', label='Validation')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'bo', label='Training')
plt.plot(epochs, val_acc, 'ro', label='Validation')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Оценка модели на тестовых данных
loss, acc = model.evaluate(X_test, y_test)
print(f'\nTest loss = {loss}\nTest accuracy = {acc}\n')

actual_labels = np.argmax(y_test, axis=1)
prediction = np.argmax(model.predict(X_test), axis=1)
print(classification_report(actual_labels, prediction))

# Матрица ошибок для тестовых данных
cm = confusion_matrix(actual_labels, prediction)
columns = ['pred_' + name for name in class_names]
indexes = ['actual_' + name for name in class_names]
cm = pd.DataFrame(cm, columns=columns, index=indexes)

sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Reds)
plt.yticks(rotation=0)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Результаты обучения модели на сбалансированных по классам данных:

# Test loss = 0.7876015901565552
# Test accuracy = 0.7658402323722839

#               precision    recall  f1-score   support
#
#            0       0.85      0.79      0.82       121
#            1       0.71      0.74      0.73       121
#            2       0.74      0.76      0.75       121
#
#     accuracy                           0.77       363
#    macro avg       0.77      0.77      0.77       363
# weighted avg       0.77      0.77      0.77       363
