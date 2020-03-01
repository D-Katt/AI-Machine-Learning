import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

# Загружаем данные из csv-файла:
data = pd.read_csv('sms.csv',
                   header=None,
                   names=['category', 'message'],
                   sep='\t')

# Смотрим на формат данных:
print(data.head())

# Проверяем соотношение категорий в выборке:
print(data.category.value_counts(normalize=True))

# Преобразуем текстовые категории ('ham' / 'spam')
# в бинарные числовые значения (1 - спам, 0 - не спам):
data['category'] = data.category.map({'ham': 0, 'spam': 1})

# Смотрим на типичную длину сообщения (количество слов):
data['length'] = data.message.str.len()

print('\nМинимальная длина сообщения:', data.length.min())
print('Максимальная длина сообщения:', data.length.max())
print('Средняя длина сообщения:', data.length.mean())
print('Медиана длины сообщения:', data.length.median())

# Делим данные на учебную и тестовую выборки:
test_data = data.sample(frac=0.2)
train_data = data.drop(test_data.index)

# Отделяем категории в отдельные переменные:
test_target = test_data.pop('category').values
train_target = train_data.pop('category').values

# Создаем словарь всех слов, встречающихся в учебных сообщениях:
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data.message.tolist())
word_index = tokenizer.word_index

# Общее количество слов в словаре:
num_words = len(word_index)

# Преобразуем учебные и тестовые сообщения в список числовых последовательностей
# В каждой последовательности 61 элемент, лишние слова отсекаем с конца,
# короткие сообщения дополняем также с конца.

train_sequences = tokenizer.texts_to_sequences(train_data.message.tolist())
train_sequences = pad_sequences(train_sequences,
                                padding='post',
                                truncating='post',
                                maxlen=61)

test_sequences = tokenizer.texts_to_sequences(test_data.message.tolist())
test_sequences = pad_sequences(test_sequences,
                               padding='post',
                               truncating='post',
                               maxlen=61)

# Создаем датасеты из полученных числовых последовательностей и меток:
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_target))
test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_target))

# Перемешиваем данные и разбиваем на пакеты по 150 экземпляров:
train_dataset = train_dataset.shuffle(int(len(data) * 0.8)).batch(150)
test_dataset = test_dataset.shuffle(int(len(data) * 0.2)).batch(150)

# Создаем модель:
model = Sequential([Embedding(num_words+100, 16),  # вектор из 16 измерений для каждого слова
                    GlobalAveragePooling1D(),  # усредняем и преобразуем в одномерный
                    Dense(16, activation="relu"),  # слой из 16 нейронов
                    Dense(1, activation="sigmoid")])  # последний слой из 1 нейрона

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Обучаем модель:
model.fit(train_dataset,
          epochs=20,
          verbose=2)

# Оцениваем точность модели на тестовых данных:
loss, accuracy = model.evaluate(test_dataset)

print('\nLoss on test data', loss)
print('Accuracy on test data:', accuracy)
