"""Модель принимает на вход одно изображение, осуществляет feature extraction
и выдает полученный одномерный массив. Для экономии памяти исходные значения
типа float32 преобразуются в значения типа float16.
Полученный массив параметров для всех сравниваемых изображений сохраняется
и может использоваться для построения матрицы сходства всех изображений со всеми,
для многократных сравнений между разными парами изображений
или для поиска в массиве наиболее похожих изображений.
"""

import tensorflow as tf

import numpy as np
import pandas as pd

from scipy.spatial import distance_matrix
from scipy.spatial import cKDTree

# Параметры TensorFlow
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16
IMG_SIZE = 299

# Готовая обученная модель классификации без финальных слоев
feature_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet',
                                                                            include_top=False,
                                                                            input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                                            pooling='avg')

# К готовой модели добавляется новый слой без функции активации,
# преобразующий выходные данные модели в тип float16:
model = tf.keras.models.Sequential([feature_model,
                                    tf.keras.layers.Layer(1536, dtype='float16', name='Dtype')])

# Сохранение конфигурации модели в файл
tf.keras.utils.plot_model(model, 'single_input_model.png', show_shapes=True)
model.summary()


def process_path(file_path: str):
    """Функция преобразует путь к файлу в обработанный массив."""
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return tf.keras.applications.inception_resnet_v2.preprocess_input(img)


def configure_for_performance(ds):
    """Функция разбивает входные данные на batches
    и использует механизм prefetch() для искорения
    обработки данных."""
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


# Список путей к исходным файлам преобразуется в TF Dataset
# (список может включать десятки тысяч снимков).
file_names = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '1_copy.jpg']
paths = np.array(file_names)
input_data = tf.data.Dataset.from_tensor_slices(paths)

# Пути к файлам преобразуются в массивы входных данных для модели:
input_data = input_data.map(process_path, num_parallel_calls=AUTOTUNE)
input_data = configure_for_performance(input_data)

# Преобразование исходных изображений в одномерные массивы (feature extraction).
# Получаемый массив имеет форму n_samples x 1,536 columns.
image_features = feature_model.predict(input_data)

# Изображения могут попарно сравниваться в произвольном порядке.
# Фичи, соответствующие изображению, извлекаются по его индексу в массиве.
idx_1 = [0, 0, 0, 1, 3]
idx_2 = [1, 2, 4, 3, 2]
features_1 = image_features[idx_1]
features_2 = image_features[idx_2]

# Два массива сравниваемых изображений проходят через слой Dot
# для получения значений cosine similarity (значения, близкие к 1,
# означают максимальное сходство, 0 - максимальная непохожесть):
cosine_model = tf.keras.layers.Dot(axes=1, normalize=True, dtype='float16')
similarity_scores = cosine_model([features_1, features_2]).numpy()

print('Сравниваемые изображения:')
print(paths[idx_1])
print(paths[idx_2])
print('Cosine similarity scores:\n', similarity_scores)

# На основе всех значений image_features можно создать матрицу расстояний
# (cosine distances) всех изображений со всеми. В этом случае максимально
# схожие изображения имеют расстояние, длизкое к нулю.
distance_matrix = distance_matrix(image_features, image_features)
distance_matrix = pd.DataFrame(distance_matrix, index=file_names, columns=file_names)

print('Матрица расстояний:')
print(distance_matrix)

# Индексирование всего массива для последующего поиска наиболее схожих изображений:
tree = cKDTree(image_features)

# Поиск пар изображений, расстояние между которыми не превышает 0.1:
image_pairs = tree.query_pairs(r=0.1)

print('Индексы схожих изображений:')
print(image_pairs)
