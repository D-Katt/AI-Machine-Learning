# Пример загрузки данных из массивов NumPy в tf.data.Dataset.
# Используется датасет 'load_wine' из библиотеки 'sklearn'.
# Создается модель с регуляризацией весов.
# Для автоматической остановки процесса обучения
# используется инструмент callbacks.EarlyStopping.

import tensorflow as tf

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

data = load_wine()

X = data.data
y = data.target

# Приводим значения в массиве X к диапазону от 0 до 1:
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

# Разбиваем данные на 3 группы - для обучения, валидации и тестирования:
train_examples, test_examples, train_labels, test_labels = train_test_split(X, y,
                                                                            test_size=0.4)
test_examples, val_examples, test_labels, val_labels = train_test_split(test_examples,
                                                                            test_labels,
                                                                            test_size=0.5)

# Передаем учебные и тестовые данные кортежами для формирования датасетов:
train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_examples, val_labels))

# Перемешиваем значения и разбиваем датасеты на пакеты по 20 элементов:
batch_size = 20
shuffle_buffer_size = 50

train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
test_dataset = test_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
val_dataset = val_dataset.shuffle(shuffle_buffer_size).batch(batch_size)

# Создаем модель:
model = tf.keras.Sequential([
                tf.keras.layers.Dense(13,
                                      kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                      activation='relu',
                                      input_shape=[13]),
                tf.keras.layers.Dense(39,
                                      kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                      activation='relu'),
                tf.keras.layers.Dense(3, activation='softmax')
                ])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Будем отслеживать точность модели в процессе обучения,
# чтобы остановить процесс, когда точность начнет снижаться:
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=5)

model.summary()  # Выводим характеристики модели

# Обучаем модель, используя для валидации отдельную группу данных.
# Обучение будет автоматически прервано при снижении точности.
history = model.fit(train_dataset,
                    epochs=30,
                    validation_data=val_dataset,
                    verbose=2,
                    callbacks=[early_stop])


def plot_history(histories, key='sparse_categorical_accuracy'):
    """Функция создает график с динамикой точности модели
    на учебных и на тестовых данных."""
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Validation')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Training')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    plt.show()


# Выводим график точности в процессе обучения модели:
plot_history([('Model', history)])

# Проверяем точность модели на группе тестовых данных,
# которые не использовались в процессе обучения:
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print('\nТочность модели на тестовых данных:', test_acc)
