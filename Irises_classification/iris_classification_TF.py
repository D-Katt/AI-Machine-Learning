# Пример классификации данных из базы 'load_iris' библиотеки 'sklearn'.
# Модель включает несколько полносвязных слоев с разными функциями активации
# (relu, selu) и регуляризацией весов и финальный слой с функцией softmax.
# Для автоматической остановки процесса обучения
# используется инструмент callbacks.EarlyStopping.

import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# Настройки отображения графиков:
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = 9, 6
plt.rcParams.update({'font.size': 11})

# Исходные данные:
data = load_iris()

X = data.data  # (150, 4)
y = data.target  # (150,)

# Приводим значения в массиве X к диапазону от 0 до 1:
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

# Разбиваем данные на 3 группы - для обучения, валидации и тестирования:
train_examples, test_examples, train_labels, test_labels = train_test_split(X, y,
                                                                            test_size=0.4,
                                                                            shuffle=True)
test_examples, val_examples, test_labels, val_labels = train_test_split(test_examples,
                                                                        test_labels,
                                                                        test_size=0.5,
                                                                        shuffle=True)

# Передаем учебные и тестовые данные кортежами для формирования датасетов:
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_examples, val_labels))

# Разбиваем датасеты на пакеты:
batch_size = 4  # Маленький объем исходных данных
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)

# Создаем модель:
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(128,
                              activation='selu',
                              input_shape=[4]),
        tf.keras.layers.Dense(64,
                              kernel_regularizer=tf.keras.regularizers.l2(0.001),
                              activation='relu'),
        tf.keras.layers.Dense(32,
                              kernel_regularizer=tf.keras.regularizers.l2(0.001),
                              activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ]
)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Будем отслеживать val_loss в процессе обучения,
# чтобы остановить процесс при отсутствии улучшений:
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=5,  # макс. кол-во эпох без улучшений
                                              restore_best_weights=True)

model.summary()  # Выводим характеристики модели

# Обучаем модель, используя для валидации отдельную группу данных.
history = model.fit(train_dataset,
                    epochs=100,
                    validation_data=val_dataset,
                    verbose=2,
                    callbacks=[early_stop])

# Основные показатели в процессе обучения модели
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Визуализация результатов:
x_axis = range(1, len(acc) + 1)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

ax1.plot(x_axis, loss, 'bo', label='Training')
ax1.plot(x_axis, val_loss, 'ro', label='Validation')
ax1.set_title('Training and validation loss')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(x_axis, acc, 'bo', label='Training')
ax2.plot(x_axis, val_acc, 'ro', label='Validation')
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.savefig('nn_training.png')
plt.show()

# Проверяем точность модели на группе тестовых данных,
# которые не использовались в процессе обучения:
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print('\nТочность модели на тестовых данных:', test_acc)

# Ошибки классификации на тестовой выборке:
pred_labels = np.argmax(model.predict(test_dataset), axis=1)
conf_matrix = confusion_matrix(test_labels, pred_labels)

target_names = data.target_names
columns = ['pred_' + name for name in target_names]
indexes = ['actual_' + name for name in target_names]
conf_matrix = pd.DataFrame(conf_matrix, columns=columns, index=indexes)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues)
plt.yticks(rotation=0)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('nn_confusion.png')
plt.show()

print(classification_report(test_labels, pred_labels))
