import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dropout, Flatten, Concatenate, Reshape, LSTM, RepeatVector, Dense, Embedding
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

class MHCNN_Poisson:
    def __init__(self, forecast_range, look_back, n_features, to_embed, X_train_embs):
        self.forecast_range = forecast_range
        self.look_back = look_back
        self.n_features = n_features
        self.to_embed = to_embed
        self.X_train_embs = X_train_embs
        self.model = self._build_model()
        self.history = None

    def _build_model(self):
        input_layer = Input(shape=(self.look_back, self.n_features))
        head_list = []
        for _ in range(self.n_features):
            conv_layer_head = Conv1D(filters=4, kernel_size=4, activation='relu')(input_layer)
            drop1 = Dropout(.2)(conv_layer_head)
            conv_layer_flatten = Flatten()(drop1)
            head_list.append(conv_layer_flatten)
        
        concat_cnn = Concatenate(axis=1)(head_list)
        reshape = Reshape((head_list[0].shape[1], self.n_features))(concat_cnn)
        lstm = LSTM(100, activation='relu')(reshape)
        repeat = RepeatVector(self.forecast_range)(lstm)
        lstm_2 = LSTM(100, activation='relu', return_sequences=True)(repeat)
        dropout = Dropout(0.2)(lstm_2)
        flat = Flatten()(dropout)
        
        inputs = []
        outputs = []
        for idx, c in enumerate(self.to_embed):
            num_unique_vals = len(np.unique(self.X_train_embs[:, idx]))
            embed_dim = int(min(np.ceil(num_unique_vals/2), 50))
            inp = Input(shape=(1,))
            out = Embedding(num_unique_vals, embed_dim, name=c)(inp)
            out = Reshape(target_shape=(embed_dim,))(out)
            inputs.append(inp)
            outputs.append(out)
        
        x = Concatenate()(outputs)
        conc_list = [flat, x]
        concat_flat = Concatenate()(conc_list)
        dense1 = Dense(128, activation='relu')(concat_flat)
        drop = Dropout(.2)(dense1)
        dense2 = Dense(16, activation='relu')(drop)
        outs = Dense(self.forecast_range, activation='relu')(dense2)
        outs = tfp.layers.IndependentPoisson(event_shape=(self.forecast_range,))(outs)
        
        model_ = Model([input_layer, inputs], outs)
        return model_

    def compile_model(self, learning_rate=0.0001):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=lambda y, model: -model.log_prob(y),
                           metrics=['mse', 'mae'])

    def fit(self, X_train, X_train_le, y_train, epochs, batch_size=100, shuffle=True, validation_split=0.2, verbose=1):
        self.history = self.model.fit([X_train, X_train_le], y_train, 
                                      epochs=epochs,
                                      shuffle=shuffle, 
                                      batch_size=batch_size,
                                      validation_split=validation_split,
                                      verbose=verbose)
        return self.history

    def predict(self, X_test, X_test_le):
        return self.model.predict([X_test, X_test_le])

    def evaluate(self, X_test, X_test_le, y_test):
        return self.model.evaluate([X_test, X_test_le], y_test)

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    def plot_metrics(self):
        metrics = [m for m in self.history.history.keys() if not m.startswith('val_')]
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 6*len(metrics)), sharex=True)
        for i, metric in enumerate(metrics):
            axes[i].plot(self.history.history[metric], label=f'Training {metric}')
            axes[i].plot(self.history.history[f'val_{metric}'], label=f'Validation {metric}')
            axes[i].set_title(f'Model {metric}')
            axes[i].set_ylabel(metric)
            axes[i].legend()
        axes[-1].set_xlabel('Epoch')
        plt.tight_layout()
        plt.show()

    def save_model(self, filepath):
        self.model.save(filepath)

    @classmethod
    def load_model(cls, filepath):
        return tf.keras.models.load_model(filepath)

    def summary(self):
        return self.model.summary()

