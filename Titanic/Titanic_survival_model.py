# Анализ данных из списка пассажиров Титаника. Модель оценки вероятности выживания пассажира.

import pandas as pd
import tensorflow as tf

data = pd.read_csv('Titanic.csv')

# Создаем числовой столбец для идентификации пола:
data['sex_bin'] = 0
data.loc[data.sex == 'female', 'sex_bin'] = 1

# Создаем числовой столбец для идентификации класса
# (заменяем исходные текстовые значения и преобразуем в число):
data['class_n'] = data['class'].str.replace('First', '1')\
                                .replace('Second', '2')\
                                .replace('Third', '3')\
                                .astype(int)

# Создаем числовой столбец для идентификации того, путешествует пассажир один или нет:
data['alone_bin'] = 0
data.loc[data.alone == 'y', 'alone_bin'] = 1

# Делаем предварительную оценку корреляции между выживаемостью и другими параметрами:
print('Sex correlation:', data.survived.corr(data.sex_bin))
print('Age correlation:', data.survived.corr(data.age))
print('Class correlation:', data.survived.corr(data.class_n))
print('Fare correlation:', data.survived.corr(data.fare))
print('Parch correlation:', data.survived.corr(data.parch))
print('Alone correlation:', data.survived.corr(data.alone_bin))
print('N of companions correlation:', data.survived.corr(data.n_siblings_spouses))

# Отбираем для модели столбцы с числовыми параметрами:
data_selected = pd.DataFrame(data[['survived', 'sex_bin', 'class_n', 'fare', 'alone_bin']])

# Нормалируем данные в столбцах, содержащих не бинарные значения
# (делим фактическое значение на максимум в соответствующем столбце,
# таким образом, все входные данные будут находиться в диапазоне между 0 и 1):
data_selected.loc[:, 'class_n'] = data_selected['class_n'] / data_selected.class_n.max()
data_selected.loc[:, 'fare'] = data_selected['fare'] / data_selected.fare.max()

# Делим данные на учебные и тестовые в соотношении 80% / 20%:
train_data = data_selected.sample(frac=0.8)
test_data = data_selected.drop(train_data.index)

# Отделяем прогнозируемые значения от входных параметров и преобразуем в массивы:
train_labels = train_data.pop('survived').to_numpy()
test_labels = test_data.pop('survived').to_numpy()

train_data = train_data.to_numpy()
test_data = test_data.to_numpy()

# Указываем характеристики слоев нейронной сети
# (в первом слое - 5 нейронов - по количеству значений во входных данных,
# далее - два слоя по 128 нейронов, последний слой - с 1 нейроном):
model = tf.keras.Sequential([tf.keras.layers.Dense(4, input_shape=[4]),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(64, activation='relu'),
                             tf.keras.layers.Dense(1, activation='sigmoid')])

# Указываем параметры модели:
model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

# Общая информация о модели:
model.summary()

# Обущаем модель на тренировочных данных, совершая 20 итераций:
model.fit(train_data, train_labels, epochs=10)

# Вычисляем точность модели на тестовых данных:
test_loss, test_accuracy = model.evaluate(test_data, test_labels)

# Выводим результат:
print('\nTest Loss {}, Test Accuracy {}\n'.format(test_loss, test_accuracy))

# Делаем прогноз по всему массиву тестовых данных:
predictions = model.predict(test_data)

# Список для расшифровки значений в базе данных:
outcome = ['Died', 'Survived']

# Выводим сравнение прогноза и фактических значений по нескольким тестовым данным:
for i in range(20):
    print('Predicted survival: {:.2%}'.format(predictions[i][0]), 'Actual outcome:', outcome[test_labels[i]])
