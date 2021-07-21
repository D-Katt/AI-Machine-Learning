"""Model for estimating car price based on categorical and numeric features.
Original data includes numeric and categorical features for cars in a csv format:
a total of over 171,000 samples belonging to 90 brands.
Neural network is used to predict car price based on it's features:
brand, model, type of combustion engine, production year and availability of automatic transmission.
keras preprocessing layers are used for scaling down numeric data and one-hot encoding categorical data.
Optionally, sample weights could be applied during training to take into account
disproportional representation of various types of cars in the dataset.
"""

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

# Plots display settings
plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams.update({'font.size': 14})

# Original data
FILE_PATH = '../input/brazilian-vehicle-prices-july-2021-fipe/tabela_fipe.csv'

# Tensorflow settings
BATCH_SIZE = 512
EPOCHS = 1000
STOP_PATIENCE = 10
WEIGHTED_SAMPLES = False

# ------------------------------ Functions --------------------------------------


def get_data():
    """Function loads and cleans the data.
    :return: DataFrame with car features and prices.
    """
    data = pd.read_csv(FILE_PATH)
    # Replace 'Zero KM' by year 2022 assuming it's a new car
    data['Ano'] = data['Ano'].str.replace('Zero KM', '2021').replace('2022', '2021')
    data['Ano'] = data['Ano'].astype(int)
    data['Automático'] = data['Automático'].astype(int)
    return data


def df_to_dataset(df: pd.DataFrame, shuffle=True, weighted=False, batch_size=32):
    """Function transforms a pd.DataFrame into tf.data.Dataset.
    :param df: Original DataFrame with price and car features.
    :param shuffle: Boolean argument specifying if the data should be shuffled.
    :param weighted: Boolean argument specifying if the dataset should contain sample weights
    :param batch_size: Batch size
    :return: Dataset for neural network
    """
    labels = df.pop('Valor')

    if weighted:  # Weight sample according to frequency of it's combustion type
        weights_combustion = {'Gasolina': 1, 'Diesel': 3.9, 'Álcool': 39.}
        weights = df['Combustível'].apply(lambda x: weights_combustion[x])
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels, weights))
    else:
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))

    ds = ds.batch(batch_size)

    return ds


def get_normalization_layer(name: str, ds: tf.data.Dataset, weighted=False):
    """Function creates a normalization layer for the specified numeric feature.
    :param name: Name of the numeric column (feature)
    :param ds: Tensorflow Dataset object containing x and y values
    :param weighted: Boolean argument specifying if the dataset contains sample weights
    :return: Normalization layer adapted to the feature scale
    """
    # Normalization layer for the feature
    normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=None)

    # Dataset that only yields specified feature
    if weighted:
        feature_ds = ds.map(lambda x, y, w: x[name])
    else:
        feature_ds = ds.map(lambda x, y: x[name])

    # Adapt the layer to the data scale
    normalizer.adapt(feature_ds)

    return normalizer


def get_category_encoding_layer(name: str, ds: tf.data.Dataset, dtype: str, max_tokens=None, weighted=False):
    """Function creates category encoding layers
    with string or integer lookup index.
    :param name: Name of the categorical feature
    :param ds: Tensorflow Dataset object containing x and y values
    :param dtype: String describing data type of the categorical feature (one of 'string' or 'int64')
    :param max_tokens: Maximum number of tokens in the lookup index
    :param weighted: Boolean argument specifying if the dataset contains sample weights
    :return: Lambda function with categorical encoding layers and lookup index
    """
    # Lookup layer which turns strings or integers into integer indices
    if dtype == 'string':
        index = tf.keras.layers.experimental.preprocessing.StringLookup(max_tokens=max_tokens)
    else:  # 'int64'
        index = tf.keras.layers.experimental.preprocessing.IntegerLookup(max_tokens=max_tokens)

    # Dataset that only yields specified feature
    if weighted:
        feature_ds = ds.map(lambda x, y, w: x[name])
    else:
        feature_ds = ds.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index
    index.adapt(feature_ds)

    # Create a discretization for integer indices
    encoder = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=index.vocab_size())

    # Apply one-hot encoding to the indices. The lambda function captures the
    # layer and index so they could be used later in the functional model.
    return lambda feature: encoder(index(feature))


def plot_history(hist):
    """Function plots a chart with training and validation metrics.
    :param hist: Tensorflow history object from model.fit()
    """
    # Losses
    mae = hist.history['loss']
    val_mae = hist.history['val_loss']

    # Epochs to plot along x axis
    x_axis = range(1, len(mae) + 1)

    plt.plot(x_axis, mae, 'bo', label='Training')
    plt.plot(x_axis, val_mae, 'ro', label='Validation')
    plt.title('Training and validation MAE')
    plt.ylabel('Loss (MAE)')
    plt.xlabel('Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------- EDA --------------------------------------

data = get_data()
print(f'DataFrame shape: {data.shape}')

print(data.head())

print(data.describe())

# Correlation of features with the target
correlation = data.corr()
ax = sns.heatmap(correlation, annot=True, cmap=plt.cm.Reds, fmt='0.3f')
l, r = ax.get_ylim()
ax.set_ylim(l + 0.5, r - 0.5)
plt.yticks(rotation=0)
plt.title('Correlation Matrix')
plt.show()

# Distribution of numeric features and target values
data.hist(bins=20, figsize=(14, 10))
plt.show()

# Distribution of categorical features
for feature in ('Marca', 'Modelo', 'Combustível'):
    print('-' * 40)
    print(f'{feature}: {data[feature].nunique()} unique values')
    groups = data[feature].value_counts(normalize=True)[:10]
    plt.barh(groups.index, groups.values)
    plt.title(f'Largest groups in {feature}')
    plt.show()

# Price correlation with year of production
plt.scatter(data['Ano'], data['Valor'])
plt.title('Price Correlation with Production Year')
plt.xlabel('Year of production')
plt.ylabel('Car price')
plt.show()

# Price depending on availability of automatic transmission
automat_price = data.groupby(by='Automático')['Valor'].mean()
print('Price premium for automatic transmission:', automat_price.max() / automat_price.min())
plt.bar(['No-automatic transmission', 'Automatic transmission'], automat_price.values)
plt.title('Average prices')
plt.show()

# Price depending on combustion type
combustion_price = data.groupby(by='Combustível')['Valor'].mean()
combustion_premiums = combustion_price / combustion_price.min()
for c_type, premium in zip(combustion_price.index, combustion_premiums):
    print(f'{c_type}: {premium}')
plt.bar(combustion_price.index, combustion_price.values)
plt.title('Average prices')
plt.show()

# Price range for all brands and models
min_price = data['Valor'].min()
max_price = data['Valor'].max()
mean_price = data['Valor'].mean()
median_price = data['Valor'].median()

print(f'Price range: {min_price} - {max_price}')
print(f'Mean price: {mean_price}\nMedian price: {median_price}')

# Max/min price ratio for car brands
brands_price_ratio = data.groupby(by='Marca')['Valor'].agg(['min', 'max', 'count'])
brands_price_ratio['price_ratio'] = brands_price_ratio['max'] / brands_price_ratio['min']
min_ratio = brands_price_ratio['price_ratio'].min()
max_ratio = brands_price_ratio['price_ratio'].max()
mean_ratio = brands_price_ratio['price_ratio'].mean()
median_ratio = brands_price_ratio['price_ratio'].median()
print(f'Max/min price ratio: {min_ratio} - {max_ratio}\nAverage price ratio = {mean_ratio}'
      f'\nMedian price ratio = {median_ratio}')
brands_price_ratio['price_ratio'].hist(bins=20)
plt.title('Max/min Price Ratio for Car Brands')
plt.xlabel('Price ratio')
plt.ylabel('Frequency')
plt.show()

# Brands with large spread between minimum and maximum price
print(brands_price_ratio[brands_price_ratio['price_ratio'] >= 50])

# Brands with small spread between minimum and maximum price
print(brands_price_ratio[brands_price_ratio['price_ratio'] <= 5])

# --------------------- Input Data for Neural Network -----------------------------

# Split original DataFrame into train, validation and test parts
# proportionally distributing car models between the groups
train, test = train_test_split(data, test_size=0.1,
                               stratify=data['Marca'], random_state=0)
train, val = train_test_split(train, test_size=0.2,
                              stratify=train['Marca'], random_state=0)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# Create Dataset objects
train_ds = df_to_dataset(train, weighted=WEIGHTED_SAMPLES, batch_size=BATCH_SIZE)
val_ds = df_to_dataset(val, shuffle=False, weighted=WEIGHTED_SAMPLES, batch_size=BATCH_SIZE)
test_ds = df_to_dataset(test, shuffle=False, batch_size=BATCH_SIZE)

# -------------------------- Neural Network ---------------------------------

# Lists to append numeric and categorical input features
all_inputs = []  # Input layers
encoded_features = []  # Results of preprocessing

# Input layer for 1 numeric value (car age)
numeric_col = tf.keras.Input(shape=(1,), name='Ano')
# Normalization layer
normalization_layer = get_normalization_layer('Ano', train_ds,
                                              weighted=WEIGHTED_SAMPLES)
# Scaled down input value
encoded_numeric_col = normalization_layer(numeric_col)
# Add the objects to lists
all_inputs.append(numeric_col)
encoded_features.append(encoded_numeric_col)

# Create layers for categorical encoding
categorical_cols = ['Marca', 'Modelo', 'Combustível']
for feature in categorical_cols:
    # Input layer for 1 string value
    categorical_col = tf.keras.Input(shape=(1,), name=feature, dtype='string')
    # Index and encode the value
    encoding_layer = get_category_encoding_layer(feature, train_ds, dtype='string',
                                                 weighted=WEIGHTED_SAMPLES)
    encoded_categorical_col = encoding_layer(categorical_col)
    # Add the objects to lists
    all_inputs.append(categorical_col)
    encoded_features.append(encoded_categorical_col)

# Add input layer for a binary feature describing automatic / no-automatic type
numeric_col = tf.keras.Input(shape=(1,), name='Automático')
all_inputs.append(numeric_col)
encoded_features.append(numeric_col)

# Create a model
all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(
    128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)
)(all_features)
x = tf.keras.layers.Dense(
    64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)
)(x)
x = tf.keras.layers.Dense(
    32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)
)(x)
output = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(all_inputs, output)

model.compile(optimizer='adam', loss='mae')

# Visualize the model graph
tf.keras.utils.plot_model(model, show_shapes=True, rankdir='LR')

model.summary()

# Train the model
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=STOP_PATIENCE,
                                              restore_best_weights=True)

history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds,
                    verbose=2, callbacks=[early_stop],
                    use_multiprocessing=True, workers=-1)

plot_history(history)

# Evaluate the model on the test set
mae = model.evaluate(test_ds)
print(f'Test MAE = {mae}')

print(f'Overall mean price: {mean_price}\nOverall median price: {median_price}')
print(f'MAE / Mean price: {mae/mean_price}')
print(f'MAE / Median price: {mae/median_price}')
