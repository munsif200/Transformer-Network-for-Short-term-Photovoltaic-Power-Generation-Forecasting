import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras import layers

plt.rcParams["font.family"] = "Times New Roman"

training_df = pd.read_csv(r"Data\Training_78-Site_2-eco-Kinetics.csv", header=0, infer_datetime_format=True, parse_dates=['timestamp'], index_col=['timestamp'])

## Pre-processing
training_df = training_df.fillna(0)

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

training_df = transform_data(training_df)
training_df = split_dataset(training_df)
Xtrain, Ytrain = to_supervised(training_df, 12)

print('shape of Training_input:', Xtrain.shape)
print('shape of Training_output:', Ytrain.shape)

# Training Parameters
MAX_EPOCHS = 200

def compile_and_fit(model, xtrain=Xtrain, ytrain=Ytrain):
    model.compile(loss=[tf.keras.losses.MeanSquaredError()],
                  optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsolutePercentageError(), tf.keras.metrics.MeanAbsoluteError()])
    
    history = model.fit(xtrain, ytrain, epochs=MAX_EPOCHS,
                        batch_size=512, validation_split=0.3, verbose=1)
    return history

def Loss(train_loss, valid_loss):
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.rcParams["figure.figsize"] = (15, 3)
    plt.title('Model Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper left')
    plt.savefig('out/loss_plot.png')
    plt.show()

def errors(Train_RMSE, Valid_RMSE, train_MAPE, Valid_MAPE):
    plt.plot(Train_RMSE)
    plt.plot(Valid_RMSE)
    plt.plot(train_MAPE)
    plt.plot(Valid_MAPE)
    plt.title('Model Errors')
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend(['Train RMSE', 'Validation RMSE', 'Train MAPE', 'Validation MAPE'], loc='upper left')
    plt.rcParams["figure.figsize"] = (15, 3)
    plt.savefig('out/errors_plot.png')
    plt.show()

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization()(inputs)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu", padding="same")(x)
    x = layers.Conv1D(filters=128, kernel_size=2, activation="relu", padding="same")(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    res = x + inputs
    norm_x = layers.LayerNormalization()(res)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(norm_x, norm_x)
    res = x + inputs
    norm_x = layers.LayerNormalization()(res)
    return norm_x

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    
    for _ in range(num_transformer_blocks):
        enc_out = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(enc_out, enc_out)
    res = x + enc_out
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = layers.Dense(832, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(mlp_dropout)(x)

    outputs = layers.Dense(12)(x)
    
    return tf.keras.Model(inputs, outputs)

input_shape = (12, 6)
CTNET = build_model(input_shape, head_size=4, num_heads=3, ff_dim=32, num_transformer_blocks=3, mlp_units=[256], mlp_dropout=0.3, dropout=0.2)

CTNET.summary()
history = compile_and_fit(CTNET)
Loss(history.history['loss'], history.history['val_loss'])
#Loss(history.history['loss'], history.history['val_loss'])

CTNET.save('out/CTNET.h5')
