#  Tensorflow.js Performance on LSTM Model Inference

### Network Construction

seed: a random slice (length = sampleLen) of Nietzsche.txt from https://storage.googleapis.com/tfjs-examples/lstm-text-generation/data/nietzsche.txt


Model: a simple LSTM model (with a little modification) from TensorFlow.js Example: https://github.com/tensorflow/tfjs-examples/tree/master/lstm-text-generation. The example is inspired by the LSTM text generation example from Keras: https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py


## URL Params

1. backend
2. lstmLayerSizes (separated by commas if multiple layers. e.g., 128 or 100,50)
3. sampleLen (optional, default 40)
4. sampleStep (optional, default 3)
5. generateLength (optional, default 200)
6. temperature (optional, default 0.75)

## Message

`jslib inference task=LSTMTextGeneration lib=tensorflowjs backend=backend layersizes=lstmLayerSizes sampleLen=sampleLen sampleStep=sampleStep generateLength=generateLength inferenceTimePerChar=avgTime`