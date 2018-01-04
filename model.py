import csv

import numpy as np

import tensorflow as tf

import cv2

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Input, Cropping2D
from keras.callbacks import ModelCheckpoint, EarlyStopping


my_csv_path = 'my_data/driving_log.csv'
udacity_csv_path = 'data/driving_log.csv'

def load(csv_filename, win):
    """
    win: whether windows style path
    """
    lines = []
    with open(csv_filename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # print(line)
            lines.append(line)
    # print(lines[0], lines[1])
    lines = lines[1:]
    center_db, left_db, right_db, steer_db = [], [], [], []
    # read csv file
    for line in lines:
        front_img_path = line[0]
        left_img_path = line[1]
        right_img_path = line[2]
        if win:
            front_filename = front_img_path.split('\\')[-1]
            left_filename = left_img_path.split('\\')[-1]
            right_filename = right_img_path.split('\\')[-1]
            front_img_path = './my_data/IMG/' + front_filename
            left_img_path = './my_data/IMG/' + left_filename
            right_img_path = './my_data/IMG/' + right_filename
        else:
            front_filename = front_img_path.split('/')[-1]
            left_filename = left_img_path.split('/')[-1]
            right_filename = right_img_path.split('/')[-1]

            front_img_path = './data/IMG/' + front_filename
            left_img_path = './data/IMG/' + left_filename
            right_img_path = './data/IMG/' + right_filename

        steering = float(line[3])
        if steering != 0.0:
            center_db.append(front_img_path)
            left_db.append(left_img_path)
            right_db.append(right_img_path)
            steer_db.append(steering)
        else:
            prob = np.random.uniform()
            if prob <= 0.2:
                center_db.append(front_img_path)
                left_db.append(left_img_path)
                right_db.append(right_img_path)
                steer_db.append(steering)
    return center_db, left_db, right_db, steer_db


center_db, left_db, right_db, steer_db = load(udacity_csv_path, False)

# shuffle the dataset
center_db, left_db, right_db, steer_db = shuffle(center_db, left_db, right_db, steer_db)

# train & valid data split
img_train, img_valid, steer_train, steer_valid = train_test_split(center_db, steer_db, test_size=0.1, random_state=42)

print(len(img_train), len(img_valid))

print(img_train[0])

def select_center_img(center, steer, index):
    image, steering = cv2.imread(center[index]), steer[index]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, steering

def select_left_img(left, steer, index, offset=0.22):
    image, steering = cv2.imread(left[index]), steer[index] + offset
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, steering

def select_right_img(right, steer, index, offset=0.22):
    image, steering = cv2.imread(right[index]), steer[index] - offset
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, steering

def translate_img(image, steer):
    """
    randomly translate image horizantally, make corresponding 
    adjustment in the steering angle also
    """
    max_shift = 55
    max_ang = 0.14  # ang_per_pixel = 0.0025

    rows, cols, _ = image.shape

    random_shift = np.random.randint(-max_shift, max_shift + 1)
    dst_steer = steer + (random_shift / max_shift) * max_ang
    if abs(dst_steer) > 1:
        dst_steer = -1 if (dst_steer < 0) else 1

    mat = np.float32([[1, 0, random_shift], [0, 1, 0]])
    dst_img = cv2.warpAffine(image, mat, (cols, rows))
    return dst_img, dst_steer

def brightness_img(image):
    """
    randomly change brightness
    """
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random = np.random.randint(2)
    if random == 0:
        random_bright = 0.2 + np.random.uniform(0.2, 0.6)
        hsv_img[:, :, 2] = hsv_img[:, :, 2] * random_bright
    op_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return op_img

def flip_img(image, steering):
    """ flip the image"""
    flip_image, flip_steering = cv2.flip(image, 1), -steering
    return flip_image, flip_steering

def augment_img(image, steering):
    image, steering = translate_img(image, steering)
    image, steering = flip_img(image, steering)
    image = brightness_img(image)
    return image, steering

def generate_valid(img_valid, steer_valid):
    """ generate validation set """
    img_set = []
    steer_set = []

    for i in range(len(img_valid)):
        img, steer = select_center_img(img_valid, steer_valid, i)
        img_set.append(img)
        steer_set.append(steer)
    return np.array(img_set), np.array(steer_set)

def generate_train_data(center, left, right, steering, data_size):
    """ generate training set """
    image_set = []
    steering_set = []

    for _ in range(data_size):
        i = np.random.randint(len(steer_train))
        random = np.random.randint(7)
        if (random== 0):
            img, steer = select_center_img(center, steering, i)
        if (random== 1):
            img, steer = select_left_img(left, steering, i)
        if (random== 2):
            img, steer = select_right_img(right, steering, i)
        if (random== 3):
            img, steer = select_center_img(center, steering, i)
            img, steer = translate_img(img, steer)
        if (random== 4):
            img, steer = select_center_img(center, steering, i)
            img = brightness_img(img)
        if (random== 5):
            img, steer = select_center_img(center, steering, i)
            img, steer = flip_img(img, steer)
        if (random ==6):
            img, steer = select_center_img(center, steering, i)
            img, steer = augment_img(img, steer)
        
        image_set.append(img)
        steering_set.append(steer)
    return np.array(image_set), np.array(steering_set)


X_train, y_train = generate_train_data(center_db, left_db, right_db, steer_db, 20480)
X_val, y_val = generate_valid(img_valid, steer_valid)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

def model():
    """
    Model inspired from nvidia self driving car
    """
    # Initializing the model
    model = Sequential()
    # Cropping layer
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160, 320, 3)))
    # Input and normalization layer
    model.add(Lambda(lambda x: x / 127.5 - 1.0))
    # First conv layer with relu activation
    model.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2), activation='relu', name='Conv1'))
    # First pooling layer: max pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
    # Second conv layer
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2), activation='relu', name='Conv2'))
    # Second pooling: max pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
    # Third conv layer
    model.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(1, 1), activation='relu', name='Conv3'))
    # Third pooling: max pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='same'))
    # Fourth conv layer
    model.add(Convolution2D(128, 2, 2, border_mode='same', subsample=(1, 1), activation='relu', name='Conv4'))
    # Flattened layers
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu', name='FC1'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', name='FC2'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', name='FC3'))
    model.add(Dense(1))
#     model.summary()
    return model

model = model()

# Callback to save model file every epoch
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
# early stopping to prevent over fitting
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='mse')

model.fit(X_train, y_train, nb_epoch=20, validation_data=(X_val, y_val), shuffle=True, verbose=1, callbacks=[checkpoint, early_stop])

