"""Pretrained DistilBERT model is initialized for classification task
with one class and compiled with regression metrics to produce readability scores.
The model is fine-tuned with decreasing learning rate on the train samples
until validation RMSE stops improving.
Texts are not preprocessed in any way before passing them to the tokenizer of cased model.
When splitting data length of text samples is taken into account to observe 
proportional distribution of short, medium and long excerpts in train and validation sets.
"""

import pandas as pd
import numpy as np
import random
import os
import gc

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Plots display settings
plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams.update({'font.size': 14})

# Seed everything
seed_value = 5
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Paths to original data
train_path = '../input/commonlitreadabilityprize/train.csv'
test_path = '../input/commonlitreadabilityprize/test.csv'

# Pretrained model
model_path = '../input/huggingface-bert-variants/distilbert-base-cased/distilbert-base-cased'

# TensorFlow settings
BATCH_SIZE = 32
EPOCHS = 20
START_LR = 1e-5
END_LR = 1e-7
PATIENCE = 3

# NLP model settings
MAX_LEN = 256  # Number of words per text

# Portion of data for validation
VAL_SIZE = 0.1

# ------------------------------ Functions --------------------------------


def get_data(path: str) -> pd.DataFrame:
    """Function loads data from the csv file
    and creates a column with total number of characters.
    :param path: Path to csv file with original data
    :return: DataFrame with original data and generated features
    """
    df = pd.read_csv(path)
    df['n_chars'] = df['excerpt'].apply(len)

    return df


def tokenize_texts(tokenizer, texts: pd.Series, labels):
    """Function converts texts into tokenized and batched datasets for the model.
    Returns dataset with or without labels.
    :param tokenizer: Tokenizer instance from transformers library
    :param texts: pd.Series with raw texts
    :param labels: Series of target values or None, if labels are not available
    :return: Tensorflow Dataset object
    """
    if labels is not None:  # Data is labeled
        ds = tokenizer(texts.values.tolist(),
                       return_tensors='tf', max_length=MAX_LEN,
                       padding='max_length', truncation=True)
        ds = tf.data.Dataset.from_tensor_slices(
            (ds['input_ids'], ds['attention_mask'], labels.values)
        ).map(lambda x1, x2, y: ({'input_ids': x1, 'attention_mask': x2}, y))\
            .batch(BATCH_SIZE)

    else:  # If no labels are provided
        ds = tokenizer(texts.values.tolist(),
                       return_tensors='tf', max_length=MAX_LEN,
                       padding='max_length', truncation=True)
        ds = tf.data.Dataset.from_tensor_slices(
            (ds['input_ids'], ds['attention_mask'])
        ).map(lambda x1, x2: {'input_ids': x1, 'attention_mask': x2})\
            .batch(BATCH_SIZE)

    return ds


def train_and_forecast() -> np.array:
    """Function fine-tunes pretrained transformer model
    and produces a forecast for the test data.
    :returns: Numpy array with readability scores for test data
    """
    # Initialize pretrained model for classification task with 1 class
    model = TFAutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Tokenize test texts and convert into tensorflow Dataset object
    test_ds = tokenize_texts(tokenizer, data_test['excerpt'], labels=None)
    print(f'Number of test text samples: {len(data_test)}')
    print(f'Test texts tokenized. Number of batches: {len(test_ds)}')

    # Split the samples of various text length proportionally
    # between train and validation sets
    groups = pd.cut(data_train['n_chars'],
                    bins=[0, 800, 900, 1000, 1100, 1200, 1350],
                    labels=[1, 2, 3, 4, 5, 6])

    train_texts, val_texts, train_scores, val_scores = train_test_split(
        data_train['excerpt'], data_train['target'], stratify=groups,
        test_size=VAL_SIZE, shuffle=True, random_state=0)

    # Tokenize train and validation texts
    train_ds = tokenize_texts(tokenizer, train_texts, train_scores)
    valid_ds = tokenize_texts(tokenizer, val_texts, val_scores)

    print(f'Number of train text samples: {len(train_texts)}')
    print(f'Train texts tokenized. Number of batches: {len(train_ds)}')

    print(f'Number of validation text samples: {len(val_texts)}')
    print(f'Validation texts tokenized. Number of batches: {len(valid_ds)}')

    # Linearly decreasing learning rate
    lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=START_LR,
        end_learning_rate=END_LR,
        decay_steps=EPOCHS * len(train_ds)
    )

    # Compile the model with regression metrics
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr_scheduler, clipnorm=1.0),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # To stop training when per-epoch validation RMSE starts growing
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error',
                                                  patience=PATIENCE,
                                                  restore_best_weights=True)

    history = model.fit(train_ds, validation_data=valid_ds,
                        epochs=EPOCHS, verbose=2, callbacks=[early_stop],
                        use_multiprocessing=True, workers=2)
    plot_history(history)

    loss, rmse = model.evaluate(valid_ds)
    print(f'Training completed. Validation loss (MSE) = {loss}\nValidation RMSE: {rmse}')

    # Make a forecast for test data
    forecast = model.predict(test_ds).logits.flatten()

    gc.collect()
    tf.keras.backend.clear_session()

    return forecast


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


# ----------------------------- Training the model --------------------------

# Load train and test data
data_train = get_data(train_path)
data_test = get_data(test_path)

# Train neural network and produce a forecast for the test set
data_test['target'] = train_and_forecast()

# Save the result
data_test[['id', 'target']].to_csv('submission.csv', index=False)
print(data_test[['id', 'target']].head())
