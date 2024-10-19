
#entity_embedding.py>

#functions


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras import layers

class DataAnalysisEmbedding:
    def __init__(self):
        plt.style.use('ggplot')
        sns.set_theme(style='darkgrid')
        plt.rcParams["figure.figsize"] = (15, 10)
        self.model = None
        self.encoders = None
        self.embs = None
        self.labels_encod = None

    @staticmethod
    def determina_stagione(data):
        giorno, mese, anno = [int(i) for i in data.split('/')]
        if (mese == 3 and giorno >= 21) or (mese == 4) or (mese == 5) or (mese == 6 and giorno < 21):
            return "Primavera"
        elif (mese == 6 and giorno >= 21) or (mese == 7) or (mese == 8) or (mese == 9 and giorno < 21):
            return "Estate"
        elif (mese == 9 and giorno >= 21) or (mese == 10) or (mese == 11) or (mese == 12 and giorno < 21):
            return "Autunno"
        else:
            return "Inverno"

    def prepare_for_encoding(self, to_embed, train):
        self.encoders = []
        X_train_le = []
        for i in range(len(to_embed)):
            le = LabelEncoder()
            transformed_column = le.fit_transform(train[to_embed[i]])
            X_train_le.append(transformed_column)
            self.encoders.append(le)
        X_train = np.array(X_train_le).T
        y_train = train.target.values.astype(np.float32)
        return X_train_le, X_train, y_train

    def build_EEmodel(self, to_embed, X_train, hidden_layers, neurons):
        inputs = []
        outputs = []
        for idx, c in enumerate(to_embed):
            num_unique_vals = len(np.unique(X_train[:, idx]))
            embed_dim = int(min(np.ceil(num_unique_vals/2), 50))
            inp = layers.Input(shape=(1,))
            out = layers.Embedding(num_unique_vals, embed_dim, name=c)(inp)
            out = layers.Reshape(target_shape=(embed_dim,))(out)
            inputs.append(inp)
            outputs.append(out)
        x = layers.Concatenate()(outputs)
        for ind_layer in range(hidden_layers):
            x = layers.Dense(neurons[ind_layer], activation='relu')(x)
        y = layers.Dense(1)(x)
        self.model = Model(inputs=inputs, outputs=y)
        return self.model

    def create_embeddings_textlabels(self, to_embed):
        if self.model is None or self.encoders is None:
            raise ValueError("Model or encoders not initialized. Call build_EEmodel and prepare_for_encoding first.")
        self.embs = []
        for id in range(len(to_embed)):
            self.embs.append(self.model.layers[id + len(to_embed)].get_weights()[0])
        self.labels_encod = []
        for i, emb in enumerate(self.embs):
            self.labels_encod.append(list(self.encoders[i].inverse_transform(range(emb.shape[0]))))
        return self.embs, self.labels_encod

    def plot_embs_j(self, j, perplexity=5, learning_rate=100, n_clusters=4):
        if self.embs is None or self.labels_encod is None:
            raise ValueError("Embeddings or labels not created. Call create_embeddings_textlabels first.")
        tsne = manifold.TSNE(init='pca', random_state=100, method='exact', perplexity=perplexity, learning_rate=learning_rate)
        Y = tsne.fit_transform(self.embs[j])
        plt.figure(figsize=(40, 30))
        if self.embs[j].shape[0] >= 10:
            km = KMeans(n_clusters=n_clusters, random_state=0)
            km.fit(self.embs[j])
            clus = km.labels_
            scatter = plt.scatter(-Y[:, 0], -Y[:, 1], c=clus, s=150)
            plt.colorbar(scatter)
        else:
            plt.scatter(-Y[:, 0], -Y[:, 1])
        for i, txt in enumerate(self.labels_encod[j]):
            plt.annotate(txt, (-Y[i, 0], -Y[i, 1]), xytext=(-20, 8), textcoords='offset points', fontsize=27)
        plt.title(f'Embedding Visualization for Feature {j}')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.show()

    def compile_model(self, optimizer='adam', loss='mse'):
        if self.model is None:
            raise ValueError("Model not initialized. Call build_EEmodel first.")
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit_model(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
        if self.model is None:
            raise ValueError("Model not initialized or compiled. Call build_EEmodel and compile_model first.")
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return history

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model not initialized or trained. Call build_EEmodel, compile_model, and fit_model first.")
        return self.model.predict(X_test)

    def plot_loss(self, history):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("Model not initialized. Cannot save.")
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)

