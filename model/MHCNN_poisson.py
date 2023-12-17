import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Input, Flatten, Activation, Reshape, RepeatVector, Concatenate, GRU, Embedding, Lambda


import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers

def MHCNN_poisson():
  input_layer = Input(shape=(LOOK_BACK, n_features)) #n_features corrispondono al numero di regioni
  head_list = []
  for i in range(0, n_features):
    conv_layer_head = Conv1D(filters=4, kernel_size=4, activation='relu')(input_layer)
    drop1 = Dropout(.2) (conv_layer_head)
    conv_layer_flatten = Flatten()(drop1)
    head_list.append(conv_layer_flatten)

  concat_cnn = Concatenate(axis=1)(head_list)
  reshape = Reshape((head_list[0].shape[1], n_features))(concat_cnn)
  lstm = LSTM(100, activation='relu')(reshape)
  repeat = RepeatVector(FORECAST_RANGE)(lstm)
  lstm_2 = LSTM(100, activation='relu', return_sequences=True)(repeat)
  dropout = Dropout(0.2)(lstm_2)
  flat = Flatten()(dropout)

  inputs = []
  outputs = []
  for idx, c in enumerate(to_embed):
    num_unique_vals = len(np.unique(X_train_embs[:, idx]))
    embed_dim = int(min(np.ceil(num_unique_vals/2), 50))
    inp = Input(shape = (1, ))
    out = Embedding(num_unique_vals, embed_dim, name = c )(inp)
    out = Reshape(target_shape=(embed_dim, ))(out)
    inputs.append(inp)
    outputs.append(out)

  x = Concatenate()(outputs)
  conc_list = [flat, x]
  concat_flat = Concatenate()(conc_list)
  dense1 = Dense(128, activation = 'relu')(concat_flat)
  drop = Dropout(.2)(dense1)
  dense2 = Dense(16, activation = 'relu')(drop)
  outs = Dense(1, activation = 'relu')(dense2)
  outs = tfpl.IndependentPoisson(1)(outs)

  model_ = Model([input_layer, [inputs]], outs)
  return model_

poisson = MHCNN_poisson()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
poisson.compile(optimizer = optimizer, 
              loss=lambda y, model: -model.log_prob(y),
              metrics=[])
poisson.fit([X_train, X_train_le], y_train, epochs = EPOCHS,
                              shuffle = True, batch_size =  100)
