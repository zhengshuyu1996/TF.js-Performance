#  JS Library Performance Comparison

we chose about seven popular neuron network JavaScript library to run training and inference task on mnist dataset.

### Network Construction

Dataset: mnist dataset on http://yann.lecun.com/exdb/mnist/

Input: 60000 images * 28 * 28 pixels * 3 channels

Model: a simple DNN model (with a little modification) in [Keras Official Examples](https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py)

Structure: 

- 2 hidden layers each with 512 units
- Activation function: relu
- Optimizer: sgd (or library’s default optimizer)
- Loss function: cross entropy 
- **No dropout layers**



