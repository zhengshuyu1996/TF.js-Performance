import argparse
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time

from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist

BATCH_SIZE = 64

IMAGE_LENGTH = 28
INPUT_NODE = 784
OUTPUT_NODE = 10
NUM_CHANNELS = 1
LEARNING_RATE = 0.15

# data, split between train and validate sets
(x_train, y_train), (x_eval, y_eval) = mnist.load_data("mnist")
x_train = x_train.reshape(x_train.shape[0], INPUT_NODE)
#print(x_train.shape)
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

def train(num, size, train_size):
    train_time = 0.0
    iter = int(train_size / BATCH_SIZE)
    #print(iter)

    for i in range(iter):
        x = x_train[i*64:(i+1)*64]
        y = y_train[i:i+1]
        print(i)
        start = time.time()
        model.fit(
            x, 
            y,
            epochs=1,
            batch_size=BATCH_SIZE,
            verbose=0
        )
        end = time.time()
        train_time = train_time + end - start

    #print(int(train_time))
    f = open("benchmark.txt", "a")
    f.write("jslib\ttrain\tmnist\tpython\tcpu\t%d\t%d\tcpu\t%f\n" % (num, size, train_time * 1000 / train_size))
    # modify this message when using CUDA
    # time s => ms
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gernerate pretrained DNN Mnist model given hidden layers' number and size.")
    parser.add_argument("-n", "--num", type=int, required=True, help="Number of hidden layers")
    parser.add_argument("-s", "--size", type=int, required=True, help="Size of hidden layers")
    parser.add_argument("-t", "--train", type=int, required=True, help="Number of training pictures")
    args = parser.parse_args()
    print("hidden layer num: " + str(args.num))
    print("hidden layer size: " + str(args.size))
    print("train size: " + str(args.train))
    print("start training...")
    init(args.num, args.size)
    train(args.num, args.size, args.train)
