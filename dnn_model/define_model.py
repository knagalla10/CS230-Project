"""Define hypermodel"""

import tensorflow as tf
import keras_tuner as kt
from keras.models import Model
from keras.layers import Input, Add, Activation, BatchNormalization, Dense, Lambda
from keras.regularizers import L2
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsolutePercentageError

tf.get_logger().setLevel('ERROR')

def input_length(X_train):
    global X_length
    X_length = X_train.shape[-1]

def pad_layer(layer, pad):
    padded_layer = tf.pad(layer, tf.constant([[0, 0], [0, pad]]), mode="constant")
    return padded_layer

class MyHyperModel(kt.HyperModel):

    def build(self, hp):
        # Input layer
        inputs = Input(shape=(X_length,), dtype="float64", name="input")

        # L2 regularizer for hidden layer weights
        lambd_2 = hp.Float("lambda_2", min_value=0.00, max_value=0.50, sampling="linear", default=0.00)

        # dense_1 layer
        units_1 = hp.Int("units_1", min_value=32, max_value=1024, step=32, default=32) # dense_1 size
        activation_1=hp.Choice("A_1", values=['relu', 'tanh'], default='relu')
        x = Dense(units_1, activation=activation_1, dtype="float64", kernel_regularizer=L2(lambd_2), name="dense_1")(inputs)

        # Hidden blocks
        num_blocks = hp.Int('num_blocks', min_value=1, max_value=20, default=1) # number of hidden blocks
        batch_norm_flag = hp.Boolean("BN_flag", default=False) # include BN layers
        residual_flag = hp.Boolean("res_flag", default=False) # include residual connections
        for i in range(num_blocks):
            # Set block_{i} hyperparameters
            l=i+2 # block{i} hidden layer index
            units = hp.Int(f"units_{l}", min_value=32, max_value=1024, step=32, default=32) # dense_{l} size
            activation = hp.Choice(f"A_{l}", ["relu", "tanh"], default="relu") # A_{l} activation function

            # Build block_{i}
            x_skip = x # initiate residual connection
            x = Dense(units, dtype="float64", kernel_regularizer=L2(lambd_2), name=f"dense_{l}")(x) # dense_{l} layer
            if residual_flag:
                if x.shape[-1] > x_skip.shape[-1]: # perform zero-padding to match dimensions of x and x_skip
                    pad = x.shape[-1] - x_skip.shape[-1]
                    x_skip = Lambda(pad_layer, arguments={'pad':pad})(x_skip)
                elif x.shape[-1] < x_skip.shape[-1]:
                    pad = x_skip.shape[-1] - x.shape[-1]
                    x = Lambda(pad_layer, arguments={'pad':pad})(x)
                x = Add()([x, x_skip]) # implement residual connection
            if batch_norm_flag:
                x = BatchNormalization(dtype="float64", name=f"BN_{l}")(x) # BN_{l} layer
            with hp.conditional_scope(f"A_{l}", ["relu"]):
                if activation == "relu":
                    x = Activation('relu', name=f"A_{l}")(x) # A_{l} layer (=relu)
            with hp.conditional_scope(f"A_{l}", ["tanh"]):
                if activation == "tanh":
                    x = Activation('tanh', name=f"A_{l}")(x) # A_{l} layer (=tanh)

        # Output layer
        activation_out=hp.Choice("A_L", values=['relu', 'linear'], default='linear')
        outputs = Dense(3, activation=activation_out, dtype="float64", kernel_regularizer=L2(lambd_2), name="output")(x)

        # Build model
        model = Model(inputs, outputs)
        return model

    def fit(self, hp, model, X_train, Y_train, X_dev, Y_dev, callbacks=None, epochs=1, **kwargs):
        # Set training batch size and define input datasets
        batch_size = hp.Int("batch_size", min_value=16, max_value=1024, step=32, default=16)
        training_set = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(batch_size)
        dev_set = tf.data.Dataset.from_tensor_slices((X_dev, Y_dev)).batch(X_dev.shape[0])

        # Set learning rate and define optimizer
        learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling="log", default=1e-4)
        optimizer=Adam(learning_rate=learning_rate)

        # Define training loss function
        loss_fn=MeanSquaredError()

        # Define performance evaluation metrics
        perf_eval=MeanAbsolutePercentageError()
    
        # Function for the training step
        @tf.function
        def run_train_step(X, Y):
            with tf.GradientTape() as tape:
                Y_hat = model(X, training=True)
                loss = loss_fn(Y_hat, Y)
                # Add L2 regularization losses
                if model.losses:
                    loss += tf.math.add_n(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
        # Function for validation step
        @tf.function
        def run_val_step(X, Y):
            Y_hat = model(X, training=False)
            perf_eval.update_state(Y, Y_hat)

        # Assign model to callbacks
        for callback in callbacks:
            callback.set_model(model)

        # Record best values of performance evaluation metrics
        best_epoch_perf = float("inf")

        # Training loop
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")

            # Iterate through training set for training step
            for ix_train, iy_train in training_set:
                run_train_step(ix_train, iy_train)

            # Iterate through dev set for validation step
            for ix_dev, iy_dev in dev_set:
                run_val_step(ix_dev, iy_dev)

            # Calling callbacks after epoch
            epoch_perf = float(perf_eval.result().numpy())
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs={"MAPE": epoch_perf})
            perf_eval.reset_state()

            print(f"Epoch performance: {epoch_perf}")
            best_epoch_perf = min(best_epoch_perf, epoch_perf)

        model.summary()

        # Return values of performance evaluation metrics
        return best_epoch_perf
