# Пример классификации данных из базы 'load_iris' библиотеки 'sklearn'.
# Один из слоев предусматривает регуляризацию весов.
# Для автоматической остановки процесса обучения
# используется инструмент callbacks.EarlyStopping.

import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

data = load_iris()

X = data.data
y = data.target

# Приводим значения в массиве X к диапазону от 0 до 1:
min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

# Преобразуем данные в DataFrame:
df = pd.DataFrame(X)
df['y'] = y

# Значения y слабо коррелируют со значениями столбца 1:
sns.pairplot(df, y_vars='y', x_vars=[0, 1, 2, 3])
plt.show()

# Удаляем столбец 1:
df.drop(1, axis='columns', inplace=True)

# Обновляем значения X и y:
y = df.pop('y')
X = df.to_numpy()

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
shuffle_buffer_size = len(train_labels)

train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
test_dataset = test_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
val_dataset = val_dataset.shuffle(shuffle_buffer_size).batch(batch_size)

# Создаем модель:
model = tf.keras.Sequential([
                tf.keras.layers.Dense(3,
                                      activation='selu',
                                      input_shape=[3]),
                tf.keras.layers.Dense(112,
                                      kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                      activation='relu'),
                tf.keras.layers.Dense(3, activation='softmax')
                ])

model.compile(optimizer=tf.keras.optimizers.Adam(),
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
                    epochs=100,
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
