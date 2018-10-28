import argparse
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist

IMAGE_LENGTH = 28
INPUT_NODE = 784
OUTPUT_NODE = 10
NUM_CHANNELS = 1
LEARNING_RATE = 0.15

train_time = 0.0
infer_time = 0.0

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
def init(num, size):
    model.add(Dense(
        input_shape=(INPUT_NODE,),
        units=size,
        activation="relu"
    ))
   
    for i in range(num - 1):
        model.add(Dense(
            units=size,
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

def train(train_size, batch_size):
    print("start training...")
    global train_time
    iter = int(train_size / batch_size)

    for i in range(iter):
        x = x_train[i*batch_size:(i+1)*batch_size]
        y = y_train[i*batch_size:(i+1)*batch_size]
        start = time.clock()
        model.fit(
            x, 
            y,
            epochs=1,
            batch_size=batch_size,
            verbose=0
        )
        end = time.clock()
        train_time = train_time + end - start

    train_time = float(train_time) / iter


def infer(infer_size):
    global infer_time
    print("starting inference...")
    x = np.ones((1, 784), dtype=float)
    
    start = time.clock()
    for i in range(infer_size):
        model.predict(x)
        x = x + np.ones((1, 784), dtype=float)
        
    end = time.clock()
    infer_time = infer_time + end - start
    infer_time = float(infer_time) / infer_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gernerate pretrained DNN Mnist model given hidden layers' number and size.")
    parser.add_argument("-n", "--num", type=int, required=True, help="Number of hidden layers")
    parser.add_argument("-s", "--size", type=int, required=True, help="Size of hidden layers")
    parser.add_argument("-b", "--batch", type=int, required=True, help="Size of batch")
    parser.add_argument("-t", "--train", type=int, required=True, help="Number of training pictures")
    parser.add_argument("-i", "--infer", type=int, required=True, help="Number of inference pictures")
    args = parser.parse_args()
    init(args.num, args.size)
    train(args.train, args.batch)
    infer(args.infer)

    f = open("benchmark.txt", "a")
    f.write("tfjs\tdnn\tpython\t%d\t%d\t%d\t%f\tpython\tpython\t%f\n" % 
        (args.num, args.size, args.batch, train_time * 1000, infer_time * 1000))
    # modify this message when using CUDA
    # time s => ms
    f.close()
