#  JS Library Performance Comparison

we chose about seven popular neuron network JavaScript library to run training and inference task on mnist dataset.

### Network Construction

Dataset: mnist dataset on http://yann.lecun.com/exdb/mnist/

Train Input: 3200 images * 28 * 28 pixels * 1 channels

Test Input: 1000 images * 28 * 28 pixels * 1 channels

Model: a simple CNN model (with a little modification) in [Keras Official Examples](https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py)

Structure: 

* Each unit is a convolution layer followed by a pooling layer
* Pooling: max pooling
  * Padding: same

- Activation function: relu
- Optimizer: sgd (or library’s default optimizer)
- Loss function: cross entropy 
- **No dropout layers**

## URL

example: `http://localhost:8000/tfjs-cnn-mnist/tensorflowjs.html?backend=cpu&traintime=10000&infertime=1000&units=1&filters=64&batchsize=64`

1. backend
2. processtime(in ms)
3. units
4. filters
5. batchsize(for training)

## Message

`tfjs cnn backend units filters batchSize avgTrainTime LoadTime warmupTime avgInferTime`