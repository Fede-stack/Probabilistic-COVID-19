import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error


# e.g., END_DAY = '01.09.2022  00:00:00'
def get_train_test(df, target_col, data_index_col, locations_col, END_DAY):
  df_pivot = pd.pivot_table(df, values=target_col, index=[data_index_col], columns=[locations_col]) #from long to wide
  train_test_split = datetime.strptime(END_DAY, '%d.%m.%Y %H:%M:%S')
  df_train = df_pivot.loc[df_pivot.index < train_test_split]
  df_test = df_pivot.loc[df_pivot.index >= train_test_split]
  return df_pivot, df_train, df_test

def to_minmax(df_train, df_test, df_pivot):
  reg_list = [i for i in df_pivot.columns if i != 'data']
  scaler = MinMaxScaler()
  scaled_train = scaler.fit_transform(df_train[reg_list])
  scaled_test = scaler.transform(df_test[reg_list])
  return scaled_train, scaled_test

def split_sequence(sequence, look_back, forecast_horizon):
  """
  Specifically for 1-day Prediction
  """
  X = [] 
  y = []
  ind_ = []
  for i in range(len(sequence)):
    lag_end = i + look_back
    forecast_end = lag_end + forecast_horizon
    if forecast_end > len(sequence):
      break
    seq_x, seq_y = sequence[i:lag_end], sequence[lag_end:forecast_end]
    ind_.append(lag_end)
    for i in range(20):
      X.append(seq_x)
    y.append(scaler.inverse_transform(seq_y)[forecast_horizon])
  return np.array(X), np.array(y), ind_

def estrai_info(data):
  """
  Info Related to DOW and Season
  """
    giorni = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]
    giorno_settimana = giorni[data.weekday()]
    mese = data.month
    anno = data.year

    if mese in [12, 1, 2]:
        stagione = "Inverno"
    elif mese in [3, 4, 5]:
        stagione = "Primavera"
    elif mese in [6, 7, 8]:
        stagione = "Estate"
    else:
        stagione = "Autunno"

    return [giorno_settimana, mese, anno, stagione]


def create_data_forEE(df, y, ind, to_embed):
  n_reg = df.columns.shape[0]
  temp_features = np.array([np.tile(estrai_info(date), (n_reg, 1)).tolist() for date in df.iloc[ind, :].index]).reshape(-1, 4) #extract info from date
  spatial_features = np.array(np.tile(df.columns, (int(y.shape[0]/n_reg), 1))).reshape(-1)
  df_ = pd.DataFrame(np.column_stack((temp_features, spatial_features)), columns = to_embed)
  return df_

def prepare_for_encoding(to_embed, train, test):
  encoders = [] # list to save different LabelEncoder objects
  X_train_le = []
  X_test_le = []

  for i in range(len(to_embed)): # to_embed represents a list of categorical variables you want to convert to embeddings
      le = LabelEncoder()
      transformed_column_train = le.fit(train[to_embed[i]]).transform(train[to_embed[i]]) 
      transformed_column_test = le.transform(test[to_embed[i]])
      X_train_le.append(transformed_column_train)
      X_test_le.append(transformed_column_test)
      encoders.append(le)

  X_train = X_train_le.copy()
  X_train = np.array(X_train).T
  X_test = X_test_le.copy()
  X_test = np.array(X_test).T

  return X_train_le, X_test_le, X_train, X_test, encoders

def evaluate_forecast(y_test, yhat):
 mae = mean_absolute_error(y_test,yhat)
 print('mae:', mae)
 mse = mean_squared_error(y_test,yhat)
 print('mse:', mse)
 return mae, mse

def create_inputs(scaled_train, scaled_test, LOOK_BACK, FORECAST_RANGE, df_train, df_test, to_embed):
  X_train, y_train, ind_train = split_sequence(scaled_train, look_back=LOOK_BACK, forecast_horizon=FORECAST_RANGE)
  X_test, y_test, ind_test = split_sequence(scaled_test, look_back=LOOK_BACK, forecast_horizon=FORECAST_RANGE)
  y_train = y_train.reshape(...).astype(np.float32) #reshape according to the range of forecast
  y_test = y_test.reshape(...).astype(np.float32)
  train_embs = create_data_forEE(df_train, y_train, ind_train, to_embed)
  test_embs = create_data_forEE(df_test, y_test, ind_test, to_embed)

  X_train_le, X_test_le, X_train_embs, X_test_embs, encoders = prepare_for_encoding(to_embed, train_embs, test_embs)
  return X_train, y_train, X_test, y_test, X_train_le, X_test_le, X_train_embs, X_test_embs, encoders
