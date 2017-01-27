import os
import re
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

CWD = os.getcwd()
TEST_WD = os.path.join(CWD, "test")
IMG_W = 66
IMG_L = 200
N_CHANNEL = 3
HEADER = ["img_c", "img_l", "img_r", "angle", "throttle", "break", "speed"]
X_VAR = "img_c"
Y_VAR = "angle"

def load_data(cwd):
    df = pd.read_csv(os.path.join(cwd, 'driving_log.csv'), header=None, names=HEADER)
    return np.asarray(df[X_VAR]), np.asarray(df[Y_VAR])    

def oversample_data(x_train, y_train, k=2):
    indices = (np.where(abs(y_train) >= 0.25))
    for i in range(k):
        x_train = np.append(x_train, np.take(x_train, indices))
        y_train = np.append(y_train, np.take(y_train, indices))
    return x_train, y_train

def flip_data(x_train, y_train):
    orientation = np.ones(y_train.shape)
    x_train = np.append(x_train, x_train)
    y_train = np.append(y_train, y_train)
    orientation = np.append(orientation, np.multiply(-1, orientation))
    return x_train, y_train, orientation

def initialize_model():
    model = Sequential()

    model.add(BatchNormalization(input_shape=(IMG_W, IMG_L, N_CHANNEL)))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", activation="relu"))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", activation="relu"))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", activation="relu"))

    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="relu"))

    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="relu"))

    model.add(Flatten())
	
    model.add(Dense(100))

    model.add(Dropout(0.5))

    model.add(Activation('relu'))

    model.add(Dense(50))
    
    model.add(Dropout(0.5))

    model.add(Activation('relu'))

    model.add(Dense(10))

    model.add(Activation('relu'))

    model.add(Dense(1))

    model.add(Activation('tanh'))

    return model

def process_img(path, flip):
    path = re.sub("/Users/kwu/Git/sdc-behavioral-cloning", CWD, path)
    img = np.asarray(Image.open(path).resize((IMG_L, IMG_W)))
    if flip == -1:
        img = cv2.flip(img, 1)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    return yuv

def data_generator(x_train, y_train, orientation, batch_size):
    while 1:
        n = len(x_train)
        x_train, y_train = shuffle(x_train, y_train)
        for offset in range(0, n, batch_size):
            end = offset + batch_size
            batch_x = np.asarray([process_img(path, flip) for path, flip in zip(x_train[offset:end], orientation[offset:end])])
            batch_y = np.expand_dims(np.multiply(y_train[offset:end], orientation[offset:end]), axis=1)
            yield (batch_x, batch_y)

if __name__=="__main__":
    
    model = initialize_model()

    x_train, y_train = load_data(CWD)
    x_train, y_train = shuffle(x_train, y_train)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)    
    x_train, y_train = oversample_data(x_train, y_train)

    ### Generate horizontally flipped samples
    # x_train, y_train, train_orientation = flip_data(x_train, y_train)
    # x_train, y_train, train_orientation = shuffle(x_train, y_train, train_orientation)

    ### Use original dataset, no flips
    train_orientation = np.ones(y_train.shape)

    x_test, y_test = load_data(TEST_WD)

    model.compile(optimizer=Adam(lr=0.0001), loss='mse', metrics=['mean_absolute_error'])
    history = model.fit_generator(data_generator(x_train, y_train, train_orientation, batch_size=64), samples_per_epoch=len(y_train), nb_epoch=5, 
        validation_data = data_generator(x_valid, y_valid, np.ones(y_valid.shape), batch_size=64), nb_val_samples = len(y_valid))

    mse, mae = model.evaluate_generator(data_generator(x_test, y_test, np.ones(y_test.shape), batch_size=64), val_samples=len(y_test))
    print ("Test set: mean squared error - {0}".format(mse))
    print ("Test set: mean absolute error - {0}".format(mae))

    json_string = model.to_json()
    model.save_weights("model.h5")
    with open("model.json", "w") as f:
        json.dump(json_string, f)


