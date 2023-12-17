import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

class DataPreprocessor:
    def __init__(self, df, target_col, data_index_col, locations_col, END_DAY, to_embed):
        self.df = df
        self.target_col = target_col
        self.data_index_col = data_index_col
        self.locations_col = locations_col
        self.END_DAY = END_DAY
        self.to_embed = to_embed
        self.df_pivot, self.df_train, self.df_test = self.get_train_test()

    def get_train_test(self):
        df_pivot = pd.pivot_table(self.df, values=self.target_col, index=[self.data_index_col], columns=[self.locations_col])
        train_test_split = datetime.strptime(self.END_DAY, '%d.%m.%Y %H:%M:%S')
        df_train = df_pivot.loc[df_pivot.index < train_test_split]
        df_test = df_pivot.loc[df_pivot.index >= train_test_split]
        return df_pivot, df_train, df_test

    def to_minmax(self):
        reg_list = [i for i in self.df_pivot.columns if i != 'data']
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(self.df_train[reg_list])
        scaled_test = scaler.transform(self.df_test[reg_list])
        return scaled_train, scaled_test

    def split_sequence(self, sequence, look_back, forecast_horizon):
        X, y, ind_ = [], [], []
        for i in range(len(sequence)):
            lag_end = i + look_back
            forecast_end = lag_end + forecast_horizon
            if forecast_end > len(sequence):
                break
            seq_x, seq_y = sequence[i:lag_end], sequence[lag_end:forecast_end]
            ind_.append(lag_end)
            for _ in range(20):
                X.append(seq_x)
            y.append(self.scaler.inverse_transform(seq_y)[forecast_horizon])
        return np.array(X), np.array(y), ind_

    def estrai_info(self, data):
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

    def create_data_forEE(self, df, y, ind):
        n_reg = df.columns.shape[0]
        temp_features = np.array([np.tile(self.estrai_info(date), (n_reg, 1)).tolist() for date in df.iloc[ind, :].index]).reshape(-1, 4)
        spatial_features = np.array(np.tile(df.columns, (int(y.shape[0]/n_reg), 1))).reshape(-1)
        df_ = pd.DataFrame(np.column_stack((temp_features, spatial_features)), columns = self.to_embed)
        return df_

    def prepare_for_encoding(self, train, test):
        encoders = []
        X_train_le, X_test_le = [], []

        for col in self.to_embed:
            le = LabelEncoder()
            transformed_column_train = le.fit_transform(train[col])
            transformed_column_test = le.transform(test[col])
            X_train_le.append(transformed_column_train)
            X_test_le.append(transformed_column_test)
            encoders.append(le)

        X_train = np.array(X_train_le).T
        X_test = np.array(X_test_le).T
        return X_train, X_test, encoders

    def evaluate_forecast(self, y_test, yhat):
        mae = mean_absolute_error(y_test, yhat)
        mse = mean_squared_error(y_test, yhat)
        return mae, mse

    def create_inputs(self, LOOK_BACK, FORECAST_RANGE):
        scaled_train, scaled_test = self.to_minmax()
        X_train, y_train, ind_train = self.split_sequence(scaled_train, LOOK_BACK, FORECAST_RANGE)
        X_test, y_test, ind_test = self.split_sequence(scaled_test, LOOK_BACK, FORECAST_RANGE)
        train_embs = self.create_data_forEE(self.df_train, y_train, ind_train)
        test_embs = self.create_data_forEE(self.df_test, y_test, ind_test)
        X_train_embs, X_test_embs, encoders = self.prepare_for_encoding(train_embs, test_embs)
        return X_train, y_train, X_test, y_test, X_train_embs, X_test_embs, encoders
