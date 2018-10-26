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

def train():
    print("starting training...")
    train_size = 640
    train_time = 0
    iter = int(train_size / BATCH_SIZE)
    print(iter)

    for i in range(iter):
        x = x_train[i*64:(i+1)*64]
        y = y_train[i:i+1]
        model.fit(
            x_train, 
            y_train,
            epochs=1,
            batch_size=BATCH_SIZE,
            verbose=0
        )

    print("training finished...")

def infer(num, size, infer_size):
    print("starting inference...")
    print(infer_size)
    infer_time = 0 
    x = np.ones((1, 784), dtype=float)
    
    start = time.time()
    for i in range(infer_size):
        model.predict(x)
        x = x + np.ones((1, 784), dtype=float)
        
    end = time.time()
    infer_time = infer_time + end - start

    f = open("benchmark.txt", "a")
    f.write("jslib\tinfer\tmnist\tpython\tcpu\t%d\t%d\t%d\t0\tcpu\t%d\n" % (infer_size, num, size, infer_time))
    # modify this message when using CUDA
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gernerate pretrained DNN Mnist model given hidden layers' number and size.")
    parser.add_argument("-n", "--num", type=int, required=True, help="Number of hidden layers")
    parser.add_argument("-s", "--size", type=int, required=True, help="Size of hidden layers")
    parser.add_argument("-i", "--infer", type=int, required=True, help="Number of inference pictures")
    args = parser.parse_args()
    print("hidden layer num: " + str(args.num))
    print("hidden layer size: " + str(args.size))
    print("infer size: " + str(args.infer))
    init(args.num, args.size)
    train()
    infer(args.num, args.size, args.infer)
