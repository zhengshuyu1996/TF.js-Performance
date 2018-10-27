#  Tensorflow.js Performance on LSTM Model Training

### Network Construction

Training data: Nietzsche.txt from https://storage.googleapis.com/tfjs-examples/lstm-text-generation/data/nietzsche.txt


Model: a simple LSTM model (with a little modification) from TensorFlow.js Example: https://github.com/tensorflow/tfjs-examples/tree/master/lstm-text-generation. The example is inspired by the LSTM text generation example from Keras: https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py


## URL Params

1. backend
2. lstmLayerSizes (separated by commas if multiple layers. e.g., 128 or 100,50)
3. processtime (in ms, default 10000)
4. numEpochs (optional, default 1)
5. examplesPerEpoch (optional, default 2048)
6. batchSize (optional, default 128)
7. validationSplit (optional, default 0)
8. learningRate (optional, default 1e-2)
9. sampleLen (optional, default 40)
10. sampleStep (optional, default 3)

## Message

`jslib training task=LSTMTextGeneration lib=tensorflowjs backend=backend layersizes=lstmLayerSizes examplesPerEpoch=examplesPerEpoch batchSize=batchSize sampleLen=sampleLen sampleStep=sampleStep trainingTimePerEpoch=avgTime`