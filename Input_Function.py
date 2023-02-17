# -- coding:utf-8 --
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras import losses, optimizers
import scipy.io as sio
import numpy as np

def load_mat_data(file_name):
    contentsMat = sio.loadmat(file_name)
    arr_mean = np.squeeze(contentsMat['arr_mean'])
    arr_min = np.squeeze(contentsMat['arr_min'])
    arr_max = np.squeeze(contentsMat['arr_max'])
    return arr_max, arr_min, arr_mean

def min_max_test_datasets(test_data, input_factor, arr_min, arr_max, arr_mean):
    for idx_col in range(0, input_factor):
        test_data[np.isnan(test_data[:, idx_col]), idx_col] = arr_mean[idx_col]

    for idx_col in range(0, input_factor):
        max_data = arr_max[idx_col]
        min_data = arr_min[idx_col]
        test_data[:, idx_col] = (test_data[:, idx_col] - min_data) / (max_data - min_data)
    return test_data

def define_model(input_size, learning_rate):
    input_data = keras.layers.Input(shape=(input_size,))

    x = Dense(32, activation='tanh')(input_data)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_data, outputs=x)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=losses.binary_crossentropy)

    # print('Model==============')
    # model.summary()
    return model

def eval_model_DNN(input_factor, input_data, model, SavePath):
    ### Scaling Datasets
    arr_max, arr_min, arr_mean = load_mat_data(SavePath + 'Mat_Norm.mat')
    input_data = min_max_test_datasets(input_data.copy(), input_factor, arr_min, arr_max, arr_mean)

    ### Evaluation Model with DNN
    print('Evaluation with Testing')
    pred_dnn_sum = 0
    for i in range(5):
        model.load_weights(SavePath + 'best_model' + str(i+1) +'.h5')
        pred_dnn = model.predict(input_data).squeeze()
        if i == 0:
            pred_dnn_sum += 0.20166384047044594*pred_dnn
        if i == 1:
            pred_dnn_sum += 0.20164402968990952*pred_dnn
        if i == 2:
            pred_dnn_sum += 0.20148119105158632*pred_dnn
        if i == 3:
            pred_dnn_sum += 0.1985192213555717*pred_dnn
        if i == 4:
            pred_dnn_sum += 0.19669171743248673*pred_dnn

           
    
    return pred_dnn_sum