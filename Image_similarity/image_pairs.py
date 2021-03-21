"""Пример использования готовой обученной модели классификации
для оценки степени сходства между изображениями.
Модель принимает на вход пары изображений, осуществляет feature extraction
и выдает значение cosine similarity для каждой пары полученных массивов.
Может использоваться для обработки одной пары или неограниченно больших
по размеру массивов попарно сравниваемых изображений.
"""

import tensorflow as tf
import numpy as np

# Параметры TensorFlow
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16
IMG_SIZE = 299

# Готовая обученная модель классификации без финальных слоев
feature_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet',
                                                                            include_top=False,
                                                                            input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                                            pooling='avg')


def build_image_model():
    """Функция создает модель для извлечения параметров
    (feature extraction) из двух изображений и определения
    степени их сходства (cosine similarity)."""
    # Модель принимает на вход два изображения с помощью двух отдельных входных слоев:
    input1 = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input1')
    input2 = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input2')
    # Оба изображения проходят через обученную модель классификации,
    # которая преобразует каждый из низ в 1D массив из 1536 значений:
    x1 = feature_model(input1)
    x2 = feature_model(input2)
    # Полученные массивы проходят через слой Dot для вычисления cosine similarity.
    # На выходе из модели - одно значения от 0 до 1, где 0 - максимально непохожие,
    # 1 - максимально схожие изображения.
    output = tf.keras.layers.Dot(axes=1, normalize=True, name='CosineDist')([x1, x2])
    # Построение модели
    model = tf.keras.Model(inputs=[input1, input2], outputs=[output])
    return model


def process_path(file_path: str):
    """Функция преобразует путь к файлу в обработанный массив."""
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return tf.keras.applications.inception_resnet_v2.preprocess_input(img)


def sort_path_pairs(x):
    """Функция преобразует пары путей к файлам в составе TF Dataset
    в словарь с именованными входными данными для модели."""
    image_1 = process_path(x[0])
    image_2 = process_path(x[1])
    return {'input1': image_1, 'input2': image_2}


def configure_for_performance(ds):
    """Функция разбивает входные данные на batches
    и использует механизм prefetch() для искорения
    обработки данных."""
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


# Построение модели
model = build_image_model()

# Сохранение конфигурации модели в файл
tf.keras.utils.plot_model(model, 'multi_input_model.png', show_shapes=True)
model.summary()

# Пользователь через консоль вводит путь к двум сравниваемым файлам:
path_1 = input('Введите путь к изображению 1:\t')
path_2 = input('Введите путь к изображению 2:\t')

# Здесь может быть любой массив из попарно сравниваемых изображений.
# Он не ограничен по объему, т.к. модель выдает только значение
# cosine similarity, не сохраняя извлекаемые из изображений фичи,
# что экономит память.
input_data = np.array([[path_1, path_2]])

# TF Dataset из пар путей к исходным файлам:
input_data = tf.data.Dataset.from_tensor_slices(input_data)

# Массив путей к файлам преобразуется в словарь с данными для модели
input_data = input_data.map(sort_path_pairs, num_parallel_calls=AUTOTUNE)

# Ускорение обработки данных
input_data = configure_for_performance(input_data)

# Получаем значения cosine similarity для всех пар изображений
# (в данном случае - для одной пары):
similarity_scores = model.predict(input_data)
print(similarity_scores)
