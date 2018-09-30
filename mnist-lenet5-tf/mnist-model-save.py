import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
BATCH_SIZE = 64
TRAIN_SIZE = 60000
TRAIN_BATCHES = TRAIN_SIZE / BATCH_SIZE
EPOCHS = 6

IMAGE_LENGTH = 28
INPUT_NODE = 784
HIDDEN_SIZE = 512
OUTPUT_NODE = 10
NUM_CHANNELS = 1
LEARNING_RATE = 0.15

# data, split between train and validate sets
(x_train, y_train), (x_eval, y_eval) = mnist.load_data("mnist")
x_train = x_train.reshape(x_train.shape[0], INPUT_NODE)
x_eval = x_eval.reshape(x_eval.shape[0], INPUT_NODE)
x_train = x_train.astype(np.float32) / 255 - 0.5
x_eval = x_eval.astype(np.float32) / 255 - 0.5


# convert class vector to binary class matrices (one-hot representation)
y_train = keras.utils.to_categorical(y_train, OUTPUT_NODE)
y_eval = keras.utils.to_categorical(y_eval, OUTPUT_NODE)

model = Sequential();
def init():
    model.add(Dense(
        input_shape=(INPUT_NODE,),
        units=HIDDEN_SIZE,
        activation="relu"
    ))
    model.add(Dense(
        units=HIDDEN_SIZE,
        activation="relu"
    ))
    model.add(Dense(
        units=OUTPUT_NODE,
        activation="softmax"
    ))
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer="sgd",
        metrics=["accuracy"]
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
    train()
    model.save('my_model.h5')
