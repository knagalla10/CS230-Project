"""Train hypermodel"""

import pandas as pd
import keras
import keras_tuner as kt
from dnn_model.define_model import *

# Read datasets
X_train = pd.read_csv('data/training_data/X_train.csv', header=None).to_numpy().T
Y_train = pd.read_csv('data/training_data/Y_train.csv', header=None).to_numpy().T
X_dev = pd.read_csv('data/dev_data/X_dev.csv', header=None).to_numpy().T
Y_dev = pd.read_csv('data/dev_data/Y_dev.csv', header=None).to_numpy().T

# Set shape variable for input layer of model
input_length(X_train)

hp = kt.HyperParameters()

tuner = kt.RandomSearch(
    hypermodel=MyHyperModel(),
    max_trials=1,
    hyperparameters=hp,
    tune_new_entries=False,
    directory="results",
    project_name="custom_training",
    overwrite=True,
)

tuner.search(
    X_train,
    Y_train,
    X_dev,
    Y_dev,
    callbacks=[keras.callbacks.TensorBoard("./tmp/tb_logs")],
    epochs=100,
)