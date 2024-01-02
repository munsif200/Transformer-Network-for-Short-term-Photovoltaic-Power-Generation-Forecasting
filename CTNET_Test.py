import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D

plt.rcParams["font.family"] = "Times New Roman"

def transform_data(data_to_transform):
    scaler = MinMaxScaler()
    scaler = scaler.fit(data_to_transform)
    trans_data = scaler.transform(data_to_transform)
    return trans_data

def to_supervised(train, n_input, n_out=12):
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = [], []
    in_start = 0
    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_out
        if out_end <= len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])
        in_start += 1
    return np.array(X), np.array(y)

def split_dataset(data):
    train = np.array(np.split(data, len(data)/12))
    return train

def evaluate_forecasts(actual, predicted):
    scores_MSE, scores_RMSE, scores_MAE, scores_MBE, scores_MAPE = [], [], [], [], []
    for i in range(actual.shape[1]):
        MSE = mean_squared_error(actual[:, i], predicted[:, i])
        MAE = mean_absolute_error(actual[:, i], predicted[:, i])
        MAPE = mean_absolute_percentage_error(actual[:, i], predicted[:, i])
        MBE = np.mean(predicted[:, i] - actual[:, i])
        RMSE = sqrt(MSE)
        scores_MSE.append(MSE)
        scores_RMSE.append(RMSE)
        scores_MAE.append(MAE)
        scores_MBE.append(MBE)
        scores_MAPE.append(MAPE)

    s_rmse, s_MSE, s_MAE, s_MAPE, s_MBE = 0, 0, 0, 0, 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s_rmse += (actual[row, col] - predicted[row, col])**2
            s_MSE += (actual[row, col] - predicted[row, col])**2
            s_MAE += np.absolute(actual[row, col] - predicted[row, col])
            epsilon = np.finfo(np.float64).eps
            s_MAPE += (np.abs(actual[row, col] - predicted[row, col]) / np.maximum(np.abs(actual[row, col]), epsilon))
            s_MBE += (predicted[row, col] - actual[row, col])

    score_RMSE = sqrt(s_rmse / (actual.shape[0] * actual.shape[1]))
    score_MSE = (s_MSE / (actual.shape[0] * actual.shape[1]))
    score_MAE = (s_MAE / (actual.shape[0] * actual.shape[1]))
    score_MAPE = (s_MAPE / (actual.shape[0] * actual.shape[1]))
    score_MBE = (s_MBE / (actual.shape[0] * actual.shape[1]))

    print('\nScore_RMSE:\n', score_RMSE)
    print('\nScore_MSE:\n', score_MSE)
    print('\nScore_MAE:\n', score_MAE)
    print('\nScore_MAPE:\n', score_MAPE)
    print('\nScore_MBE:\n', score_MBE)

    score_df = pd.DataFrame({'RMSE': [score_RMSE],
                             'MSE': [score_MSE],
                             'MAE': [score_MAE],
                             'MAPE': [score_MAPE],
                             'MBE': [score_MBE]})

    score_df.to_csv('out/error_scores.csv', index=False)

CTNET = tf.keras.models.load_model("out/CTNET.h5")
load_dataset = pd.read_csv(r"Data\Testing_78-Site_2-eco-Kinetics.csv", header=0, infer_datetime_format=True, parse_dates=['timestamp'], index_col=['timestamp'])
testing_df = load_dataset.fillna(0)
testing_df = transform_data(testing_df)
testing_df = split_dataset(testing_df)

Xtest, Ytest = to_supervised(testing_df, 12)
CTNET_predictions = CTNET.predict(Xtest)
evaluate_forecasts(Ytest, CTNET_predictions)

st_range = 100
end = 2000
plt.plot(load_dataset.index[st_range:end], CTNET_predictions[st_range:end, 0], label='pred')
plt.plot(load_dataset.index[st_range:end], Ytest[st_range:end, 0], label='Actual')
plt.legend(loc='upper left')
plt.savefig('out/prediction_plot.png')
