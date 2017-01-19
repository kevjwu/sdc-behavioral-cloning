import os
import cv2
import json
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization

CWD = "/Users/kwu/Git/carnd-behavioral-cloning-p3"
IMG_W = 66
IMG_L = 200
N_CHANNEL = 3
HEADER = ["img_c", "img_l", "img_r", "angle", "throttle", "break", "speed"]
X_VAR = "img_c"
Y_VAR = "angle"

def load_data():
    df = pd.read_csv(os.path.join(CWD, 'driving_log.csv'), header=None, names=HEADER)
    return np.asarray(df[X_VAR]), np.asarray(df[Y_VAR])

def oversample_data(x_train, y_train, k=2):
    indices = (np.where(abs(y_train) >= 0.2))
    for i in range(k):
        x_train = np.append(x_train, np.take(x_train, indices))
        y_train = np.append(y_train, np.take(y_train, indices))
    return x_train, y_train

def load_validation(df, valid_frac=0.1):
    valid_idx = int(len(df) * valid_frac)
    x_valid = np.asarray([process_img(path) for path in df[:valid_idx]["img_c"]])
    y_valid = np.expand_dims(np.asarray(df[:valid_idx]["angle"]), axis=1)
    return x_valid, y_valid

def initialize_model():
    model = Sequential()

    model.add(BatchNormalization(input_shape=(IMG_W, IMG_L, N_CHANNEL)))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", activation="relu"))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", activation="relu"))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", activation="relu"))

    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="relu"))

    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="relu"))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(1164))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(100))

    model.add(Dense(50))

    model.add(Dense(10))

    model.add(Dense(1))

    model.add(Activation('tanh'))

    return model

def process_img(path):
    rgb = np.asarray(Image.open(path).resize((IMG_L, IMG_W)))
    yuv = cv2.cvtColor(rgb, cv2.COLOR_BGR2YCR_CB)
    return yuv

def train_generator(x_train, y_train, batch_size):
    while 1:
        n = len(x_train)
        x_train, y_train = shuffle(x_train, y_train)
        for offset in range(0, n, batch_size):
            end = offset + batch_size
            batch_x = np.asarray([process_img(path) for path in x_train[offset:end]])
            batch_y = np.expand_dims(y_train[offset:end], axis=1)
            yield (batch_x, batch_y)

if __name__=="__main__":
    
    model = initialize_model()

    x_train, y_train = load_data()
    x_train, y_train = oversample_data(x_train, y_train)
    x_train, y_train = shuffle(x_train, y_train)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)    

    model.compile(optimizer=Adam(lr=0.0001), loss='mse', metrics=['mean_absolute_error'])

    history = model.fit_generator(train_generator(x_train, y_train, batch_size=64), samples_per_epoch=len(y_train), nb_epoch=5, 
        validation_data = train_generator(x_valid, y_valid, batch_size=len(y_valid)), nb_val_samples = len(y_valid))

    json_string = model.to_json()
    model.save_weights("model.h5")
    with open("model.json", "w") as f:
        json.dump(json_string, f)
