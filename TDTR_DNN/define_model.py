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

def build_model(hp):
    # Input layer
    inputs = Input(shape=(X_length,), dtype="float32", name="input")

    # layer parameters
    activation="tanh"
    num_hidden=3
    res_block_flag=False
    
    # dense_1 layer
    x = Dense(512, activation, dtype="float32", name="dense_1")(inputs)

    # Hidden blocks
    for i in range(num_hidden):
        # Build block_i
        l=i+2 # hidden layer index
        x_skip = x # initiate residual connection
        x = Dense(512, dtype="float32", name=f"dense_{l}")(x) # dense_l layer
        if res_block_flag:
            x = Add()([x, x_skip]) # implement residual connection
        x = BatchNormalization(dtype="float32", name=f"BN_{l}")(x) # BN_l layer
        x = Activation(activation, name=f"A_{l}")(x) # A_l layer

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
tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective=kt.Objective("val_mean_absolute_percentage_error", direction="min"),
    max_trials=1,
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
    epochs=100
)

best_hp = tuner.get_best_hyperparameters()[0]
best_model = tuner.get_best_models(num_models=1)