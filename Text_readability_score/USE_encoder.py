"""Universal Sentence Encoder is used for getting text embeddings.
Text features are added to embeddings matrix and passed to a densely connected
neural network, which trains on 5 different folds of train data set.
Final prediction for the test data set is calculated as average between all predictions.
"""

import pandas as pd
import numpy as np
import string
import re
import gc
from collections import Counter

import tensorflow as tf
from tensorflow_hub import KerasLayer

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

# Plots display settings
plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams.update({'font.size': 14})

# Paths to original data
train_path = '../input/commonlitreadabilityprize/train.csv'
test_path = '../input/commonlitreadabilityprize/test.csv'

# Universal sentence encoder model
encoder_path = '../input/universalsentenceencodermodels/universal-sentence-encoder-models/use'

# TensorFlow settings
BATCH_SIZE = 64

# Number of folds for cross-validation
FOLDS = 5

# Text features
features = ['n_chars', 'n_words', 'n_sent', 'punct_signs', 'quotes',
            'quotes_per_sent', 'signs_per_sent', 'words_per_sent',
            'chars_per_sent', 'chars_per_word', 'n_vowels', 'n_digits',
            'vowels_per_word', 'vowels_ratio', 'n_unique', 'unique_ratio',
            'vowels_comb', 'consonants_comb', 'ths', 'difficult']

target = 'target'

# Regex expressions to search for combinations
# of 3 or more vowels or consonants in texts
RE_VOWELS = re.compile(r'[aeiouy]{3,}')
RE_CONSONANTS = re.compile(r'[bcdfghjklmnpqrstvwxz]{3,}')

# ---------------------------------- Functions --------------------------------------


def get_data(path: str) -> pd.DataFrame:
    """Function loads data from the csv file
    and creates features based on texts.
    :param path: Path to csv file with original data
    :return: DataFrame with original data and generated features
    """
    df = pd.read_csv(path)

    # Cleaned text (lowercase, punctuation removed)
    df['text'] = df['excerpt'].apply(preprocess_text)

    # Features based on text excerpts
    df['n_chars'] = df['text'].apply(len)  # Total number of characters without punctuation
    df['n_words'] = df['text'].apply(lambda s: len(s.split(' ')))  # Total number of words
    # Number of sentences, punctuation signs, quotes and vowels
    tmp = df['excerpt'].apply(process_characters)
    df[['n_sent', 'punct_signs', 'quotes', 'n_vowels', 'n_digits']] = tmp.apply(pd.Series)
    df['quotes_per_sent'] = df['quotes'] / df['n_sent']  # Average number of quotes per sentence
    df['signs_per_sent'] = df['punct_signs'] / df['n_sent']  # Average number of signs per sentence
    df['words_per_sent'] = df['n_words'] / df['n_sent']  # Average number of words per sentence
    df['chars_per_sent'] = df['n_chars'] / df['n_sent']  # Average number of characters per sentence
    df['chars_per_word'] = df['n_chars'] / df['n_words']
    df['vowels_per_word'] = df['n_vowels'] / df['n_words']  # Average number of vowels per word
    df['vowels_ratio'] = df['n_vowels'] / df['n_chars']  # Number of vowels to the total number of characters
    df['n_unique'] = df['text'].apply(lambda x: set(x).__len__())  # Number of unique words
    df['unique_ratio'] = df['n_unique'] / df['n_words']  # Number of unique words to total n_words
    # Sounds and combinations difficult to pronounce
    tmp = df['text'].apply(difficult_sounds)
    df[['vowels_comb', 'consonants_comb', 'ths']] = tmp.apply(pd.Series)
    df['difficult'] = df[['vowels_comb', 'consonants_comb', 'ths']].sum(axis=1)

    return df


def preprocess_text(s: str) -> str:
    """Function converts text to lowercase, removes punctuation
    and replaces multiple spaces.
    :param s: original text string
    :return: cleaned text string
    """
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub('\s+', ' ', s)
    return s.lower()


def process_characters(s: str) -> tuple:
    """Function calculates total number of punctuation signs in the text,
    number of sentences assuming any sentence contains just one '.' or '?' or '!'
    and the number of quotes in the text.
    :param s: Original text string with punctuation
    :return: Tuple with 5 int values: total number of sentences,
    total number of punctuation signs, number of quotation pairs,
    number of vowels, number of digits
    """
    chars = Counter(s)
    # Number of sentences
    sentences = 0
    for char in '.?!':
        sentences += chars[char]
    # Total number of punctuation signs
    punct_signs = 0
    for char in string.punctuation:
        punct_signs += chars[char]
    # Number of vowels
    vowels = 0
    for char in 'aeiouy':
        vowels += chars[char]
    # Number of digits
    digits = 0
    for char in '0123456789':
        digits += chars[char]
    return sentences, punct_signs, chars['"'] // 2, vowels, digits


def difficult_sounds(s: str) -> tuple:
    """Function counts number of combinations of 3 or more
    vowels and consonants and number of 'th's in the text string.
    :param s: Lowercase text string
    :return: Tuple with 3 int values (n_vowels, n_consonants, n_th)
    """
    s = s.replace(' ', '')  # Delete all spaces
    n_vowels = len(RE_VOWELS.findall(s))
    n_consonants = len(RE_CONSONANTS.findall(s))
    n_th = s.count('th')
    return n_vowels, n_consonants, n_th


def correlation_distribution(df: pd.DataFrame):
    """Function visualizes correlation between features
    and target value for the train set.
    :param df: DataFrame with original data and generated features
    """
    # Correlation heatmap
    ax = sns.heatmap(df.corr(),
                     center=0, annot=True, cmap='RdBu_r')
    l, r = ax.get_ylim()
    ax.set_ylim(l + 0.5, r - 0.5)
    plt.yticks(rotation=0)
    plt.title('Correlation matrix')
    plt.show()

    # Individual correlations between features and the target
    for feature in features:
        print(f'{feature}: {df[feature].min()} - {df[feature].max()}, '
              f'mean = {df[feature].mean()}, median = {df[feature].median()}')
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(df[feature], df[target])
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.title(f'{feature} correlation')
        plt.subplot(1, 2, 2)
        plt.hist(df[feature], bins=20)
        plt.title(f'{feature} distribution')
        plt.show()

    # Distribution of target values and standard errors
    for par in ('target', 'standard_error'):
        print(f'{par}: {df[par].min()} - {df[par].max()}, mean = {df[par].mean()}, median = {df[par].median()}')
        plt.hist(df[par], bins=20)
        plt.title(f'{par} distribution')
        plt.show()


def get_embeddings() -> tuple:
    """Function loads pretrained model from local file
    and converts train and test texts into embeddings.
    :return: Tuple with 2 np.arrays for train and test text embeddings
    Each array has the shape of n_samples x 512 features
    """
    # Initialize the model loading Universal Sentense Encoder
    # into a KerasLayer from Kaggle dataset file
    model = tf.keras.Sequential(
        [KerasLayer(encoder_path, input_shape=[], dtype=tf.string,
                    output_shape=[512], trainable=False),
        # tf.keras.layers.Layer(512, dtype=tf.float16)  # To reduce memory footprint
        ]
    )

    train_emb = model.predict(data_train['text'])
    print('Train texts converted into embeddings. Shape:', train_emb.shape)

    test_emb = model.predict(data_test['text'])
    print('Test texts converted into embeddings. Shape:', test_emb.shape)

    return train_emb, test_emb


def get_model(n_features: int):
    """Function initializes a densely connected neural network,
    which transforms input data of float values (text embeddings
    and features) into readability scores.
    :param n_features: Number of features in the input 1D array
    """
    # Initialize a new model
    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(128, input_shape=(n_features,), activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
         tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
         tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
         tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
         tf.keras.layers.Dense(1)]
    )

    # Add loss function and metrics
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(n_features: int = 512, threshold: float = 0.7):
    """Function trains a neural network on 5 folds of train data
    saving checkpoints and averaging weights across all checkpoints.
    :param n_features: Number of features in the input 1D array
    :param threshold: RMSE threshold to observe when saving the model weights
    :returns: Trained Tensorflow model for estimating readability score based on text embeddings
    """

    # Split the samples of various text length equally
    # between the folds and train several models
    # using different training and validation subsets

    fold = 0  # Counter for folds with acceptable RMSE

    for train_index, valid_index in kf.split(data_train, y=data_train['n_chars']):

        fold += 1
        exit_fold = False

        # Train neural network on this subset of data until RMSE is lower than threshold
        while not exit_fold:
            print(f'Started training on fold {fold}')

            train_x = emb_train[train_index]
            train_y = data_train.loc[train_index, 'target']

            test_x = emb_train[valid_index]
            test_y = data_train.loc[valid_index, 'target']

            train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(BATCH_SIZE)
            valid_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(BATCH_SIZE)

            # Train the model
            model = get_model(n_features=n_features)

            # To stop training when no progress is observed
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error',
                                                          patience=5,
                                                          restore_best_weights=True)

            history = model.fit(train_ds,
                                validation_data=valid_ds,
                                epochs=100,
                                verbose=2,
                                callbacks=[early_stop])

            # Loss metric on the validation set
            loss, rmse = model.evaluate(valid_ds)
            print(f'Validation loss (MSE) = {loss}\nValidation RMSE: {rmse}')

            # Save only the weights
            if rmse < threshold:

                model.save_weights(f'model_{fold}.h5')
                print(f'Saved model checkpoint for fold {fold}')

                # Visualize training progress
                plot_history(history)

                # Exit training on current fold
                exit_fold = True


def get_test_scores(n_features: int) -> pd.Series:
    """Function initializes several neural networks and makes predictions
    using saved weights. Final prediction is calculated as average value
    across all predictions.
    :param n_features: Number of features in the input 1D array
    :return: Predicted readability scores for the test set
    """
    predictions = dict()

    # Load weights from trained models and produce several predictions
    for num in range(1, 6):
        model = get_model(n_features=n_features)
        model.load_weights(f'./model_{num}.h5')
        predictions[f'target_{num}'] = model.predict(emb_test).flatten()

    # Convert all predictions to a DataFrame and calculate the average
    predictions = pd.DataFrame(predictions)
    columns = [f'target_{num}' for num in range(1, 6)]
    predictions['target'] = predictions[columns].mean(axis=1)

    return predictions['target']


def plot_history(hist):
    """Function plots a chart with training and validation metrics.
    :param hist: Tensorflow history object from model.fit()
    """
    # Losses and metrics
    mse = hist.history['loss']
    val_mse = hist.history['val_loss']
    rmse = hist.history['root_mean_squared_error']
    val_rmse = hist.history['val_root_mean_squared_error']

    # Epochs to plot along x axis
    x_axis = range(1, len(mse) + 1)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

    ax1.plot(x_axis, mse, 'bo', label='Training')
    ax1.plot(x_axis, val_mse, 'ro', label='Validation')
    ax1.set_title('Training and validation MSE')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()

    ax2.plot(x_axis, rmse, 'bo', label='Training')
    ax2.plot(x_axis, val_rmse, 'ro', label='Validation')
    ax2.set_title('Training and validation RMSE')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('RMSE')
    ax2.legend()

    plt.tight_layout()
    plt.show()


# --------------------------------- EDA & Data Processing -----------------------------------

# Load train and test data and generate text features
data_train = get_data(train_path)
data_test = get_data(test_path)
print(data_train.isna().sum())

# Correlation between all features and the target in the train set
correlation_distribution(data_train)

# ---------------------------- Preparing Input for the Model --------------------------------

# Convert texts into embeddings using Universal Sentense Encoder
emb_train, emb_test = get_embeddings()
gc.collect()
tf.keras.backend.clear_session()

# Scale text features to the range between 0 and 1 and join with text embeddings
scaler = MinMaxScaler()
emb_train = np.hstack((emb_train, scaler.fit_transform(data_train[features])))
emb_test = np.hstack((emb_test, scaler.transform(data_test[features])))

# ------------------------------ Training Neural Network -----------------------------------

# Cross-validation splitter
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=0)

# Train several neural networks on different folds of train data
# until each model achieves RMSE lower than the threshold
# and save weights of successful models to files
n_feat = len(features)
train_model(n_features=512+n_feat, threshold=0.7)
gc.collect()
tf.keras.backend.clear_session()

# ----------------------- Readability Scores for the Test Set --------------------------------

# Estimate readability scores for test texts
data_test['target'] = get_test_scores(n_features=512+n_feat)
gc.collect()
tf.keras.backend.clear_session()

# Save the result
data_test[['id', 'target']].to_csv('submission.csv', index=False)
print(data_test[['id', 'target']])
