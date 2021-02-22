"""Transfer learning is used to create a classification model
for identifying financial news sentiment from the retail investor perspective.
TF Hub pre-trained model is used as a basis with additional layers for 3 classes.
Original token based text embedding model was trained on English Google News 130GB corpus.
"""

import tensorflow_hub as hub
import tensorflow as tf

import pandas as pd
import numpy as np
import string

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# DataFrame display settings
pd.set_option('display.max_colwidth', 250)

# Charts display settings
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = 9, 6
plt.rcParams.update({'font.size': 11})

# Variable defines if classes in the original dataset will be equally balanced.
BALANCED = True

# Original data: 4,846 text samples with labelled categories
data = pd.read_csv('all_data.csv',
                   header=None,
                   names=['sentiment', 'text'],
                   encoding='latin-1',
                   dtype={'sentiment': 'category'})

# ---------------------- EDA and text preprocessing ------------------------

# Balance of classes
classes_distribution = data['sentiment'].value_counts()

# Class distribution plot
labels = classes_distribution.index
values = classes_distribution.values
plt.pie(values, labels=labels, startangle=90, autopct='%1.1f%%')
plt.title('Distribution of Classes')
plt.tight_layout()
plt.show()

# Pre-trained model takes a batch of sentences in a 1-D tensor of strings as input
# and preprocesses it by splitting text samples on spaces.
# Punctuation removal and other text cleaning operations are not included.


def text_cleaning(s: str):
    """Function removes punctuation and doubles spaces
    from the text string, removes any words that contain
    less that 2 characters and converts the string to lowercase."""
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = s.replace('  ', ' ').lower()
    s = ' '.join([word for word in s.split() if (len(word) > 1 and word.isalpha())])
    return s


data['text_cleaned'] = data['text'].apply(text_cleaning)

# Text length statistics
data['n_words'] = data['text_cleaned'].apply(lambda x: len(x.split()))

min_length = data['n_words'].min()
max_length = data['n_words'].max()
mean_length = data['n_words'].mean()
median_length = data['n_words'].median()

print(f'Sentence length: {min_length} - {max_length} words\nMean length = '
      f'{mean_length}\nMedian length = {median_length}')

# Text length distribution
plt.hist(data['n_words'], bins=10)
plt.axvline(mean_length, color='red', label='Mean')
plt.axvline(median_length, color='green', label='Median')
plt.legend()
plt.title('Headlines Length')
plt.xlabel('Number of words')
plt.ylabel('Frequency')
plt.show()

if BALANCED:
    # Number of samples in the least represented class
    quota = data['sentiment'].value_counts().min()
    # New DataFrame to add equal number of samples from each class
    balanced_data = pd.DataFrame(columns=['sentiment', 'text_cleaned'])
    # Reduce each group to the chosen number of samples
    data_groups = data.groupby('sentiment')
    for group in data_groups.indices:
        reduced_class = data_groups.get_group(group)[['sentiment', 'text_cleaned']].iloc[:quota, :]
        balanced_data = balanced_data.append(reduced_class, ignore_index=True)
    # Update the original DataFrame
    data = balanced_data
    del balanced_data

# Input data for the model as a plain text
X = data['text_cleaned']

# Categories for classification
class_names = ['negative', 'neutral', 'positive']

# Binary columns for each category
for category in class_names:
    data[category] = data['sentiment'].apply(lambda x: 1 if x == category else 0)

# Target values for the model
y = data[class_names].values

# Withhold 20% of the original data for test purposes.
# Take into account class imbalances during the split stratifying the data according to y labels.
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=1)

# Convert target values into tensors
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)

# ------------------------ Loading and modifying the model --------------------------

# Text embedding is based on Swivel co-occurrence matrix factorization with pre-built OOV.
# The model maps from text to 20-dimensional embedding vectors.
# Vocabulary contains 20,000 tokens and 1 out of vocabulary bucket for unknown tokens.

# Load the model from TF Hub
hub_layer = hub.KerasLayer('https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1',
                           output_shape=[20],
                           input_shape=[],
                           dtype=tf.string,
                           trainable=True)

# Sequential model with additional layers.
# L2 regularization is added to decrease overfitting on a small dataset.
model = tf.keras.Sequential(
    [
        hub_layer,  # Original model
        tf.keras.layers.Dense(64, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # Fully connected layer
        tf.keras.layers.Dense(32, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(3, activation='softmax')  # Final layer for 3 classes
    ]
)

model.summary()

# Metrics and optimizer
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

EPOCHS = 500

# Training will stop if no improvement is seen for 10 epochs
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                              patience=10,
                                              restore_best_weights=True)

# Dictionary for class_weight parameter of the model
# to compensate for class imbalance.
if BALANCED:
    balancing_dict = {index: 1/3 for index in range(3)}
# If we use original data with imbalanced classes,
# assign higher weights to less represented categories.
else:
    balancing_dict = {0: 0.50, 1: 0.15, 2: 0.35}

# Train the model using 20% of the training data for validation
history = model.fit(X_train, y_train,
                    batch_size=32, validation_batch_size=32,
                    validation_split=0.2, shuffle=True,
                    class_weight=balancing_dict,
                    callbacks=[early_stop],
                    epochs=EPOCHS, verbose=2)

# Evaluate training and validation progress
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training')
plt.plot(epochs, val_loss, 'ro', label='Validation')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'bo', label='Training')
plt.plot(epochs, val_acc, 'ro', label='Validation')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model on the test set
loss, acc = model.evaluate(X_test, y_test)
print(f'\nTest loss = {loss}\nTest accuracy = {acc}\n')

actual_labels = np.argmax(y_test, axis=1)
prediction = np.argmax(model.predict(X_test), axis=1)
print(classification_report(actual_labels, prediction))

# Confusion matrix for the test set
cm = confusion_matrix(actual_labels, prediction)
columns = ['pred_' + name for name in class_names]
indexes = ['actual_' + name for name in class_names]
cm = pd.DataFrame(cm, columns=columns, index=indexes)

sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Reds)
plt.yticks(rotation=0)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Results for training on a balanced dataset:

# Test loss = 0.7876015901565552
# Test accuracy = 0.7658402323722839

#               precision    recall  f1-score   support
#
#            0       0.85      0.79      0.82       121
#            1       0.71      0.74      0.73       121
#            2       0.74      0.76      0.75       121
#
#     accuracy                           0.77       363
#    macro avg       0.77      0.77      0.77       363
# weighted avg       0.77      0.77      0.77       363
