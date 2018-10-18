import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.initializers import TruncatedNormal, Zeros

BATCH_SIZE = 64
# TRAIN_BATCHES = 7500, 7500 * 64 = 8 * 60000
EPOCHS = 8
TEST_BATCH_SIZE = 1000

IMAGE_LENGTH = 28
INPUT_NODE = 784
OUTPUT_NODE = 10
NUM_CHANNELS = 1

LEARNING_RATE = 0.15
CONV1_SIZE = 5
CONV1_DEEP = 6
CONV2_SIZE = 5
CONV2_DEEP = 16
FLATTEN = 120
DENSE1_SIZE = 120
DENSE2_SIZE = 84

INPUT_SHAPE = (IMAGE_LENGTH, IMAGE_LENGTH, NUM_CHANNELS)

# data, split between train and validate sets
(x_train, y_train), (x_eval, y_eval) = mnist.load_data("mnist")
x_train = x_train.reshape(x_train.shape[0], IMAGE_LENGTH, IMAGE_LENGTH, NUM_CHANNELS)
x_eval = x_eval.reshape(x_eval.shape[0], IMAGE_LENGTH, IMAGE_LENGTH, NUM_CHANNELS)
x_train = x_train.astype(np.float32) / 255
x_eval = x_eval.astype(np.float32) / 255
x_eval = x_eval[:TEST_BATCH_SIZE]


# convert class vector to binary class matrices (one-hot representation)
y_train = keras.utils.to_categorical(y_train, OUTPUT_NODE)
y_eval = keras.utils.to_categorical(y_eval, OUTPUT_NODE)
y_eval = y_eval[:TEST_BATCH_SIZE]

model = Sequential()

def init():
    model.add(Conv2D(
        input_shape=INPUT_SHAPE,
        filters=CONV1_DEEP, 
        kernel_size=(CONV1_SIZE, CONV1_SIZE),
        kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1),
        activation='relu',
        strides=1,
        padding='same'
    ))
    
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="same"
    ))

    model.add(Conv2D(
        filters=CONV2_DEEP, 
        kernel_size=(CONV2_SIZE, CONV2_SIZE),
        kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1),
        activation='relu',
        strides=1,
        padding='same'
    ))
    
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="same"
    ))

    model.add(Flatten())

    model.add(Dense(
        units=DENSE1_SIZE,
        activation="relu",
        use_bias=True,
        bias_initializer=Zeros(),
        kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1)
    ))

    model.add(Dense(
        units=DENSE2_SIZE,
        activation="relu",
        use_bias=True,
        bias_initializer=Zeros(),
        kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1)
    ))

    model.add(Dense(
        units=OUTPUT_NODE,
        activation="softmax",
        kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1)
    ))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer='sgd',
        metrics=['accuracy']
    )

def train():
    model.fit(
        x_train, 
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2,
        validation_data=(x_eval, y_eval)
    )



if __name__ == "__main__":
    print("start training...")
    init()
    start = time.time()
    train()
    end = time.time()
    print("total training time(wall clock time): ", end-start, "s")
