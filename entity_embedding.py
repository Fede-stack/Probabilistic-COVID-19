
#entity_embedding.py>

#functions


# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# %matplotlib inline
plt.style.use('ggplot')
sns.set_theme(style = 'darkgrid')
plt.rcParams["figure.figsize"] = (15,10)

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras import layers

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

def prepare_for_encoding(to_embed, train):
  encoders = [] # list to save different LabelEncoder objects
  X_train_le = []

  for i in range(len(to_embed)):
      le = LabelEncoder()
      transformed_column = le.fit(train[to_embed[i]]).transform(train[to_embed[i]])
      X_train_le.append(transformed_column)
      encoders.append(le)

  X_train = X_train_le.copy()
  X_train = np.array(X_train).T
  y_train = train.target.values.astype(np.float32)

  return X_train_le, X_train, y_train, encoders

def EEmodel(to_embed, X_train, hidden_layers, neurons):
  inputs = []
  outputs = []
  for idx, c in enumerate(to_embed):
      num_unique_vals = len(np.unique(X_train[:, idx]))
      embed_dim = int(min(np.ceil(num_unique_vals/2), 50))
      inp = layers.Input(shape = (1, ))
      out = layers.Embedding(num_unique_vals, embed_dim, name = c )(inp)
      out = layers.Reshape(target_shape=(embed_dim, ))(out)
      inputs.append(inp)
      outputs.append(out)

  x = layers.Concatenate()(outputs)
  for ind_layer in range(hidden_layers):
    x = layers.Dense(neurons[ind_layer], activation = 'relu')(x)
  y = layers.Dense(1)(x)

  model = Model(inputs = inputs, outputs = y)
  return model

def create_embeddings_textlabels(to_embed, model, encoders):
  embs = []
  for id in range(len(to_embed)):
    embs.append(model.layers[id + len(to_embed)].get_weights()[0])

  labels_encod = []
  for i, emb in enumerate(embs):
    labels_encod.append(list(encoders[i].inverse_transform(range(emb.shape[0]))))

  return embs, labels_encod



def plot_tsne_for_j(j, embs, labels_encod):

    tsne = manifold.TSNE(init='pca', random_state=100, method='exact', perplexity=embs[j].shape[1]-1, learning_rate=100)
    Y = tsne.fit_transform(embs[j])
    if embs[j].shape[0] >= 10:
      km = KMeans(n_clusters = 4)
      km.fit(embs[j])
      clus = km.labels_
      plt.scatter(-Y[:, 0], -Y[:, 1], c = clus)
    else:
      plt.scatter(-Y[:, 0], -Y[:, 1])
    for i, txt in enumerate(labels_encod[j]):
        plt.annotate(txt, (-Y[i, 0],-Y[i, 1]), xytext = (-20, 8), textcoords = 'offset points', fontsize=27)

    num_rows = (len(embs) + 2) // 3  # Calcola il numero di righe necessarie per visualizzare tutti i plot su 3 colonne
    plt.figure(figsize=(40,30))
    for idx in range(len(embs)):
       plt.subplot(num_rows, 3, idx+1)  # crea un subplot su 3 colonne
       plot_tsne_for_j(idx, embs, labels_encod)
       plt.title(f"Plot for j={idx}")

    plt.tight_layout()
    plt.show()

def plot_embs_j(embs, labels_encod, j):
  tsne = manifold.TSNE(init='pca', random_state=100, method='exact', perplexity=5, learning_rate=100)
  Y = tsne.fit_transform(embs[j])
  plt.figure(figsize=(40,30))
  if embs[j].shape[0] >= 10:
        km = KMeans(n_clusters = 4, random_state = 0)
        km.fit(embs[j])
        clus = km.labels_
        plt.scatter(-Y[:, 0], -Y[:, 1], c = clus, s = 150)
  else:
    plt.scatter(-Y[:, 0], -Y[:, 1])
  for i, txt in enumerate(labels_encod[0]):
      plt.annotate(txt, (-Y[i, 0],-Y[i, 1]), xytext = (-20, 8), textcoords = 'offset points', fontsize=27)



