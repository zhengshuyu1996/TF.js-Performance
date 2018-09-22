import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

LEARNING_RATE = 0.15
REGULARAZTION_RATE = 0.01
BATCH_SIZE = 64
TRAIN_BATCHES = 1500
TEST_BATCH_SIZE = 1000
TEST_ITERATION_FREQUENCY = 5
IMAGE_SIZE = 28
INPUT_NODE = 784
OUTPUT_NODE = 10
NUM_CHANNELS = 1

CONV1_SIZE = 5
CONV1_DEEP = 6
CONV2_SIZE = 5
CONV2_DEEP = 16
DENSE1_SIZE = 120
DENSE2_SIZE = 84


def inference(input_tensor): #, regularizer):
    # conv1
    with tf.variable_scope("conv1"):
        conv1_weights = tf.get_variable(
                "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
    # conv1_biases = tf.get_variable(
    #            "bias", [CONV1_DEEP], 
    #            initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(
                input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding="SAME")
    #    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        relu1 = tf.nn.relu(conv1)

    # pool1
    with tf.variable_scope("pool1"):
        pool1 = tf.nn.max_pool(
                relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # conv2
    with tf.variable_scope("conv2"):
        conv2_weights = tf.get_variable(
                "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
        # conv2_biases = tf.get_variable(
        #        "bias", [CONV2_DEEP], 
        #        initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(
                pool1, conv2_weights, strides=[1, 1, 1, 1], padding="VALID")
        # relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        relus2 = tf.nn.relu(conv2)
    # pool2
    with tf.variable_scope("pool2"):
        pool2 = tf.nn.max_pool(
                relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # get pool shape
    pool_shape = pool2.get_shape().as_list()
    
    # flatten
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [tf.shape(pool2)[0], nodes])

    # dense1
    with tf.variable_scope("dense1"):
        dense1_weights = tf.get_variable(
                "weight", [nodes, DENSE1_SIZE],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
        dense1_biases = tf.get_variable(
                "bias", [DENSE1_SIZE],
                initializer=tf.constant_initializer(0.0))
        # if regularizer is not None:
        #    tf.add_to_collection("losses", regularizer(dense1_weights))
        dense1 = tf.nn.relu(
                tf.matmul(reshaped, dense1_weights) + dense1_biases)

    # dense2
    with tf.variable_scope("dense2"):
        dense2_weights = tf.get_variable(
                "weight", [DENSE1_SIZE, DENSE2_SIZE],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
        dense2_biases = tf.get_variable(
                "bias", [DENSE2_SIZE],
                initializer=tf.constant_initializer(0.0))
    # if regularizer is not None:
    #        tf.add_to_collection("losses", regularizer(dense2_weights))
        dense2 = tf.nn.relu(
                tf.matmul(dense1, dense2_weights) + dense2_biases)

    # dense3
    with tf.variable_scope("dense3"):
        dense3_weights = tf.get_variable(
                "weight", [DENSE2_SIZE, OUTPUT_NODE],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
        dense3_biases = tf.get_variable(
                "bias", [OUTPUT_NODE],
                initializer=tf.constant_initializer(0.0))
    # if regularizer is not None:
    #        tf.add_to_collection("losses", regularizer(dense3_weights))
        dense3 = tf.matmul(dense2, dense3_weights) + dense3_biases
        
    return dense3


def train(mnist):
    x = tf.placeholder(
            tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name="x-input")
    y_ = tf.placeholder(
            tf.float32, [None, OUTPUT_NODE], name="y-input")
    global_step = tf.Variable(0, trainable=False)

    # regularizer = tf.contrib.layers.l2_regularizer(
    #        REGULARAZTION_RATE)
    y = inference(x)#, regularizer)

    # wrong ?
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    #        logits=tf.cast(tf.argmax(y, 1), tf.float32),
    #        labels=tf.cast(tf.argmax(y_, 1), tf.float32))


    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits = y,
        labels = tf.argmax(y_, 1)
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean #+ tf.add_n(tf.get_collection('losses'))
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).\
        minimize(loss, global_step=global_step)

    # define validation
    correction_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        vx = mnist.validation.images

        vx_reshaped = np.reshape(vx, (len(vx),
                                      IMAGE_SIZE,
                                      IMAGE_SIZE,
                                      NUM_CHANNELS))
        validate_feed = {x: vx_reshaped,
                         y_: mnist.validation.labels}

        for i in range(TRAIN_BATCHES):
            # prepare training data
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          IMAGE_SIZE,
                                          IMAGE_SIZE,
                                          NUM_CHANNELS))
            sess.run(train_step, feed_dict={x: reshaped_xs, y_: ys})

            # check validation
            if i % TEST_ITERATION_FREQUENCY == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy is %g"
                      % (i, validate_acc))


def main(argv=None):
    mnist = input_data.read_data_sets("./mnist-data", one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()

