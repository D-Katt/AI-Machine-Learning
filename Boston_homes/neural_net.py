"""Neural network for house pricing.
Dataset: Boston homes
Model: TensorFlow densely connected neural network
"""

import os
import random
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

VAL_SIZE = 0.2
SEED = 5
BATCH_SIZE = 64
EPOCHS = 500
PATIENCE = 30

# Plots display settings
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams.update({'font.size': 14})


def set_seed(seed=42):
    """Utility function to use for reproducibility.
    :param seed: Random seed
    :return: None
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


set_seed(SEED)

# Original data
X, y = load_boston(return_X_y=True)

# Create a TF dataset that includes all available data.
ds = tf.data.Dataset.from_tensor_slices((X, y))

# Calculate number of train and validation samples.
valid_samples = int(len(X) * VAL_SIZE)
train_samples = len(X) - valid_samples

# Divide the dataset into train and validation subsets
# and apply batch().
ds_train = ds.take(train_samples).batch(BATCH_SIZE)
ds_valid = ds.skip(train_samples).take(valid_samples).batch(BATCH_SIZE)

# Initialize the normalization layer and adapt to numerical input data.
normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=None)
train_features = ds_train.map(lambda features, targets: features)
normalizer.adapt(train_features)

# Define a Sequential model that includes normalization layer.
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(X.shape[1],)),
        normalizer,
        tf.keras.layers.Dense(64, activation='selu'),
        tf.keras.layers.Dense(32, activation='selu'),
        tf.keras.layers.Dense(1)
    ]
)

# Define optimizer, loss and metrics.
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='mse',
              metrics=['mae'])

# Display the model architecture
model.summary()

# To stop the training when validation loss does not improve
# fore the specified number of epochs.
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

# To decrease learning rate when validation loss stalls.
lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.6, patience=1,
    cooldown=1, min_lr=1e-10)

# To save the best model chechpoint.
check_point = tf.keras.callbacks.ModelCheckpoint(
    './checkpoint', save_best_only=True)

# Train the model
history = model.fit(ds_train, validation_data=ds_valid,
                    epochs=EPOCHS, callbacks=[early_stop, lr, check_point])

mse, mae = model.evaluate(ds_valid)
print(f'Test loss (MSE): {mse}\nTest MAE: {mae}')
# Test loss (MSE): 21.07403564453125
# Test MAE: 3.6274964809417725

x_axis = range(len(history.history['loss']))
plt.plot(x_axis, history.history['mae'], label='Train')
plt.plot(x_axis, history.history['val_mae'], label='Validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.title('Training Progress')
plt.tight_layout()
plt.savefig('neural_net_loss.png')
plt.show()
