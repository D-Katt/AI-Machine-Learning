"""Используется 'Wine recognition dataset' из библиотеки 'sklearn'.
Создается модель с полносвязными слоями с регуляризацией весов
и входным слоем для нормализации исходных числовых данных.
Для автоматической остановки процесса обучения
используется инструмент callbacks.EarlyStopping.
"""

import tensorflow as tf
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

# Настройки отображения графиков:
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = 9, 6
plt.rcParams.update({'font.size': 11})

# Исходные данные:
data = load_wine()
X = data.data  # (178, 13)
y = data.target  # (178,)

# -------------------------- Создание модели -------------------------------

# Слой для нормализации данных, который станет составной частью модели:
normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
normalizer.adapt(X)  # Здесь определяется среднее значение и стандартное отклонение

# Модель с входным слоем, который нормализует данные при их передаче в модель
inputs = tf.keras.Input(shape=[13])
x = normalizer(inputs)
x = tf.keras.layers.Dense(32,
                          kernel_regularizer=tf.keras.regularizers.l2(0.001),
                          activation='relu')(x)
x = tf.keras.layers.Dense(16,
                          kernel_regularizer=tf.keras.regularizers.l2(0.001),
                          activation='relu')(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Параметры:
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.summary()  # Характеристики модели

# ----------------------------- Формирование датасетов ----------------------------------

# Преобразуем исходные необработанные данные в датасет:
dataset = tf.data.Dataset.from_tensor_slices((X, y))

# Перемешиваем датасет и разбиваем на пакеты
n_samples = len(X)
batch_size = 8  # С учетом маленького объема исходных данных
shuffle_buffer_size = n_samples
dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size)

# Разбиваем датасет на 3 выборки:
n_batches = n_samples // batch_size

train_size = round(0.6 * n_batches)
val_size = int(0.2 * n_batches)
test_size = int(0.2 * n_batches)

train_set = dataset.take(train_size)
val_set = dataset.skip(train_size).take(val_size)
test_set = dataset.skip(train_size).skip(val_size)

# ------------------------- Обучение модели и оценка точности ---------------------------

# Для остановки обучения модели при отсутствии улучшений в течение 5 эпох:
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=5,
                                              restore_best_weights=True)

# Обучаем модель, используя для валидации отдельную группу данных.
# Обучение будет автоматически прервано при отсутствии прогресса.
history = model.fit(train_set,
                    epochs=100,
                    validation_data=val_set,
                    verbose=2,
                    callbacks=[early_stop])

# Основные показатели в процессе обучения модели
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

x_axis = range(1, len(acc) + 1)

plt.plot(x_axis, loss, 'bo', label='Training')
plt.plot(x_axis, val_loss, 'ro', label='Validation')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('nn_loss.png')
plt.show()

plt.plot(x_axis, acc, 'bo', label='Training')
plt.plot(x_axis, val_acc, 'ro', label='Validation')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('nn_acc.png')
plt.show()

# Проверяем точность модели на группе тестовых данных,
# которые не использовались в процессе обучения:
test_loss, test_acc = model.evaluate(test_set)
print(f'\nTest accuracy: {test_acc}\nTest loss: {test_loss}')

# Сохранение модели с весами и параметрами оптимизатора:
model.save('wine_classifier.h5')
