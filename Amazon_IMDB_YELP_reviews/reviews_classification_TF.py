# Классификация отзывов с использованием нейронной сети.

import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

# Считываем данные из csv-файла:
data = pd.read_csv('reviews_data.csv',
                   header=None,
                   names=['review', 'sentiment', 'source'],
                   dtype={'source': 'category'},
                   delimiter='\t')

# Проверяем структуру и типы данных:
print(data.shape)
print(data.dtypes)
print(data.head())
print(data.groupby('source').sentiment.value_counts())

# Смотрим на типичную длину текста (количество слов):
data['length'] = data.review.str.split()
data['length'] = data['length'].apply(lambda x: len(x))
print(data.head())

print('\nМин. длина сообщения:', data.length.min())
print('Макс. длина сообщения:', data.length.max())
print('Средняя длина сообщения:', data.length.mean())
print('Медиана длины сообщения:', data.length.median())

# Для унификации текстов будем использовать лимит длины текста,
# равный средней длине + 2 станд. отклонения:
limit = int(data.length.mean() + 2 * data.length.std())
print('\nСреднее значение + 2 станд. отклонения:', limit)

# Делим данные на учебные и тестовые:
X_train, X_test, y_train, y_test = train_test_split(data.review, data.sentiment)

# Создаем словарь:
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

# Общее количество слов в словаре:
num_words = len(word_index)

# Преобразуем тексты в список числовых последовательностей
# одинаковой длины, лишние слова отсекаем с конца,
# короткие тексты дополняем также с конца.

train_sequences = tokenizer.texts_to_sequences(X_train)
train_sequences = pad_sequences(train_sequences,
                                padding='post',
                                truncating='post',
                                maxlen=limit)

test_sequences = tokenizer.texts_to_sequences(X_test)
test_sequences = pad_sequences(test_sequences,
                               padding='post',
                               truncating='post',
                               maxlen=limit)

# Создаем датасеты из полученных числовых последовательностей и меток:
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, y_test))

# Перемешиваем данные и разбиваем на пакеты по 150 экземпляров:
train_dataset = train_dataset.shuffle(len(data)).batch(150)
test_dataset = test_dataset.shuffle(len(data)).batch(150)

# Создаем модель:
model = Sequential([Embedding(num_words+1, 16),  # вектор из 16 измерений для каждого слова
                    GlobalAveragePooling1D(),  # усредняем и преобразуем в одномерный
                    Dense(16, activation="relu"),  # слой из 16 нейронов
                    Dense(1, activation="sigmoid")])  # последний слой из 1 нейрона

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Обучаем модель:
model.fit(train_dataset,
          epochs=30,
          verbose=2)

# Оцениваем точность модели на тестовых данных:
loss, accuracy = model.evaluate(test_dataset)

print('\nLoss on test data', loss)
print('Accuracy on test data:', accuracy)
