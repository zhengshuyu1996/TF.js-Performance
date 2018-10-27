import argparse
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist

IMAGE_LENGTH = 28
INPUT_NODE = 784
OUTPUT_NODE = 10
NUM_CHANNELS = 1
LEARNING_RATE = 0.15
INPUT_SHAPE = (IMAGE_LENGTH, IMAGE_LENGTH, NUM_CHANNELS)
KERNEL_SIZE = 5
POOL_SIZE = 2

train_time = 0.0
infer_time = 0.0

# data, split between train and validate sets
(x_train, y_train), (x_eval, y_eval) = mnist.load_data("mnist")
x_train = x_train.reshape(x_train.shape[0], IMAGE_LENGTH, IMAGE_LENGTH, NUM_CHANNELS)
x_eval = x_eval.reshape(x_eval.shape[0], IMAGE_LENGTH, IMAGE_LENGTH, NUM_CHANNELS)
x_train = x_train.astype(np.float32) / 255
x_eval = x_eval.astype(np.float32) / 255


# convert class vector to binary class matrices (one-hot representation)
y_train = keras.utils.to_categorical(y_train, OUTPUT_NODE)
y_eval = keras.utils.to_categorical(y_eval, OUTPUT_NODE)

model = Sequential();
def init(units, filters):
    for i in range(units):
        if i == 1:
            model.add(Conv2D(
                input_shape=INPUT_SHAPE,
                filters=filters, 
                kernel_size=KERNEL_SIZE,
                activation='relu',
                strides=1,
                padding='same'
            ))
        else:
            model.add(Conv2D(
                filters=filters, 
                kernel_size=KERNEL_SIZE,
                activation='relu',
                strides=1,
                padding='same'
            ))
        model.add(MaxPooling2D(
            pool_size=POOL_SIZE,
            strides=POOL_SIZE,
            padding="same"
        ))
    
    model.add(Flatten())

    model.add(Dense(
        units=OUTPUT_NODE,
        activation="softmax"
    ))
    
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer="sgd",
        metrics=["accuracy"]
    )

def train(train_size, batch_size):
    print("start training...")
    global train_time
    iter = int(train_size / batch_size)

    for i in range(iter):
        x = x_train[i*batch_size:(i+1)*batch_size]
        y = y_train[i*batch_size:(i+1)*batch_size]
        start = time.time()
        model.fit(
            x, 
            y,
            epochs=1,
            batch_size=batch_size,
            verbose=0
        )
        end = time.time()
        train_time = train_time + end - start

    train_time = float(train_time) / iter


def infer(infer_size):
    global infer_time
    print("starting inference...")
    x = np.ones((1, IMAGE_LENGTH, IMAGE_LENGTH, 1), dtype=float)
    
    start = time.time()
    for i in range(infer_size):
        model.predict(x)
        x = x + np.ones((1, IMAGE_LENGTH, IMAGE_LENGTH, 1), dtype=float)
        
    end = time.time()
    infer_time = infer_time + end - start
    infer_time = float(infer_time) / infer_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gernerate pretrained DNN Mnist model given hidden layers' number and size.")
    parser.add_argument("-u", "--units", type=int, required=True, help="Number of units(a conv layer + pooling layer)")
    parser.add_argument("-f", "--filters", type=int, required=True, help="Number of filters used in each conv layer")
    parser.add_argument("-b", "--batch", type=int, required=True, help="Size of batch")
    parser.add_argument("-t", "--train", type=int, required=True, help="Number of training pictures")
    parser.add_argument("-i", "--infer", type=int, required=True, help="Number of inference pictures")
    args = parser.parse_args()
    init(args.units, args.filters)
    train(args.train, args.batch)
    infer(args.infer)

    f = open("benchmark.txt", "a")
    f.write("tfjs\tcnn\tpython\t%d\t%d\t%d\t%f\tpython\tpython\t%f\n" % 
        (args.units, args.filters, args.batch, train_time * 1000, infer_time * 1000))
    # modify this message when using CUDA
    # time: s => ms
    f.close()
