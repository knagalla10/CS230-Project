"""Define hypermodel"""

import pandas as pd
import tensorflow as tf
import keras
import keras_tuner as kt
from keras.models import Model
from keras.layers import Input, Add, Activation, BatchNormalization, Dense, Lambda

tf.get_logger().setLevel('ERROR')

def input_length(X_train):
    global X_length
    X_length = X_train.shape[-1]

def pad_layer(layer, pad):
    padded_layer = tf.pad(layer, tf.constant([[0, 0], [0, pad]]), mode="constant")
    return padded_layer

def build_model(hp):
    # Input layer
    inputs = Input(shape=(X_length,), dtype="float32", name="input")

    # dense_1 layer
    x = Dense(256, activation="tanh", dtype="float32", name="dense_1")(inputs)

    # Hidden block configuration
    hidden_config = hp.Choice("hidden_config", [1, 2, 3, 4], default=1)
    with hp.conditional_scope("hidden_config", [1]):
        if hidden_config == 1:
            res_block_flag=False
            num_hidden=2
            units=64
    with hp.conditional_scope("hidden_config", [2]):
        if hidden_config == 2:
            res_block_flag=False
            num_hidden=4
            units=64
    with hp.conditional_scope("hidden_config", [3]):
        if hidden_config == 3:
            res_block_flag=False
            num_hidden=4
            units=32
    with hp.conditional_scope("hidden_config", [4]):
        if hidden_config == 4:
            res_block_flag=True
            num_hidden=4
            units=32

    # Hidden blocks
    for i in range(num_hidden):
        # Build block_i
        l=i+2 # hidden layer index
        x_skip = x # initiate residual connection
        x = Dense(units, dtype="float32", name=f"dense_{l}")(x) # dense_l layer
        if res_block_flag:
            if x.shape[-1] > x_skip.shape[-1]: # perform zero-padding to match dimensions of x and x_skip
                pad = x.shape[-1] - x_skip.shape[-1]
                x_skip = Lambda(pad_layer, arguments={'pad':pad})(x_skip)
            elif x.shape[-1] < x_skip.shape[-1]:
                pad = x_skip.shape[-1] - x.shape[-1]
                x = Lambda(pad_layer, arguments={'pad':pad})(x)
            x = Add()(x, x_skip) # implement residual connection
            x = BatchNormalization(dtype="float32", name=f"BN_{l}")(x) # BN_l layer
        x = Activation(activation="relu", name=f"A_{l}")(x) # A_l layer

    # Output layer
    outputs = Dense(3, activation="linear", dtype="float32", name="output")(x)

    # Build model
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=[keras.metrics.MeanAbsolutePercentageError()])

    return model

# Read datasets
X_train = pd.read_csv('data/training_data/X_train.csv', header=None).to_numpy().T
Y_train = pd.read_csv('data/training_data/Y_train.csv', header=None).to_numpy().T
X_dev = pd.read_csv('data/dev_data/X_dev.csv', header=None).to_numpy().T
Y_dev = pd.read_csv('data/dev_data/Y_dev.csv', header=None).to_numpy().T

# Set shape variable for input layer of model
input_length(X_train)

# Define tuner
tuner = kt.GridSearch(
    hypermodel=build_model,
    objective=kt.Objective("val_mean_absolute_percentage_error", direction="min"),
    max_trials=4,
    directory="results",
    project_name="search_architechture",
    overwrite=True,
)

# Begin search
tuner.search(
    X_train,
    Y_train,
    validation_data=(X_dev, Y_dev),
    callbacks=[keras.callbacks.TensorBoard("./tmp/tb_logs", write_graph=False)],
    epochs=250
)

best_hp = tuner.get_best_hyperparameters()[0]
best_model = tuner.get_best_models(num_models=1)