import pandas as pd
import numpy as np
import tensorflow as tf

# Define the EarlyStopping callback
early_stopping_callback_trials = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',          # Metric to monitor
    patience=10,                 # Number of epochs to wait for improvement
    verbose=1,                  # Verbosity mode (1 = progress messages)
    mode='min',                 # Use 'min' for loss metrics
    restore_best_weights=False  # Revert to the best weights after stopping
)

# Define ReduceLROnPlateau callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',     # Metric to monitor
    factor=0.5,            # Factor to reduce the learning rate by
    patience=3,            # Number of epochs with no improvement before reducing
    min_lr=1e-12           # Lower bound on the learning rate
)

def create_model(
    num_samples_in: int,
    num_channels_in: int,
    num_channels_out: int,
):
    
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape=(num_samples_in, num_channels_in)))

    # Block 0
    model.add(tf.keras.layers.Conv1D(16, 3, dilation_rate=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv1D(16, 3, dilation_rate=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv1D(16, 5, padding='same', use_bias=False))
    model.add(tf.keras.layers.AveragePooling1D(pool_size=2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # Block 1
    model.add(tf.keras.layers.Conv1D(32, 3, dilation_rate=4, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv1D(32, 3, dilation_rate=4, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv1D(32, 5, strides=2, padding='same', use_bias=False))
    model.add(tf.keras.layers.AveragePooling1D(pool_size=2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # Block 2
    model.add(tf.keras.layers.Conv1D(64, 3, dilation_rate=8, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv1D(64, 3, dilation_rate=8, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv1D(64, 5, strides=4, padding='same', use_bias=False))
    model.add(tf.keras.layers.AveragePooling1D(pool_size=2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # Fully Connected Layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(32, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_channels_out, use_bias=False))

    return model


def load_npz_to_tf_dataset(npz_path, batch_size=32, shuffle=True):
    data = np.load(npz_path)
    features = data["features"].astype("float32")   # (N, 2, 512)
    labels   = data["labels"].astype("float32")     # (N,)

    # 👉 swap axes to (N, 512, 2)
    features = np.transpose(features, (0, 2, 1))

    # Expand labels to (N, 1) so keras treats them as regression targets
    labels = labels[:, None]

    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(features), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_dataset = load_npz_to_tf_dataset("/mnt/d/Binance/high_train_data.npz") # Created by the dataset_generator
test_dataset = load_npz_to_tf_dataset("/mnt/d/Binance/high_test_data.npz") # Created by the dataset_generator

# CREATE MODELS

Model = create_model(512, 2, 1)

Model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# TRAIN MODELS

Model.fit(train_dataset, validation_data=test_dataset, epochs=100, callbacks=[early_stopping_callback_trials, reduce_lr])

# SAVE MODELS

Model.save('/mnt/d/Binance/HIGHModel.keras')