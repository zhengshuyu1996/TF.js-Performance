'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''
from __future__ import print_function
from optparse import OptionParser
from keras.callbacks import LambdaCallback
from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
from keras import layers
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import time


# Display progress logs on stdout
# logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()

op.add_option("--maxlen", type=int, dest="maxlen", default=40,
              help="")
op.add_option("--step", type=int, dest="step", default=3,
              help="")
op.add_option("--layerSizes", dest="layerSizes",
              help="")
op.add_option("--generateLength", type=int, dest="generateLength", default=200,
              help="")
op.add_option("--temperature", dest="temperature", default=0.75,
              help="")
op.add_option("--RNN", dest="rnn", default='LSTM',
              help="RNN type, support values [SimpleRNN, GRU, LSTM]")
op.add_option("--batchSize", dest="batch_size", default=128,
              help="")
op.add_option("--numEpochs", dest="numEpochs", default=1,
              help="")
op.add_option("--examplesPerEpoch", dest="examplesPerEpoch", default=2048,
              help="")

op.print_help()

argv = sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


if opts.rnn == 'SimpleRNN':
    RNN = layers.SimpleRNN
elif opts.rnn == 'GRU':
    RNN = layers.GRU
else:
    RNN = layers.LSTM

layerSizes = [int(i) for i in opts.layerSizes.split(',')]

maxlen = opts.maxlen
step = opts.step
generateLength = opts.generateLength
temperature = opts.temperature
examplesPerEpoch = opts.examplesPerEpoch


path = './Nietzsche.txt'
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


x = x[:examplesPerEpoch]
y = y[:examplesPerEpoch]

# build the model:
print('Build model...')
model = Sequential()
for i in range(len(layerSizes)):
    size = layerSizes[i]
    if i == 0:
      model.add(RNN(size, return_sequences=(i<len(layerSizes)-1), input_shape=(maxlen, len(chars))))
    else:
      model.add(RNN(size, return_sequences=(i<len(layerSizes)-1)))

model.add(layers.Dense(len(chars), activation='softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    # print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    # for diversity in [0.2, 0.5, 1.0, 1.2]:
    #     print('----- diversity:', diversity)

    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)

    for i in range(opts.generateLength):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, opts.temperature)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()

# print_callback = LambdaCallback(on_epoch_end=on_epoch_end)


# model.fit(x, y,
#           batch_size=128,
#           epochs=60,
#           callbacks=[print_callback])

print(x.shape)
time_start = time.time()
model.fit(x, y,
          batch_size=opts.batch_size,
          epochs=opts.numEpochs)
time_end = time.time()
time_training = time_end - time_start


time_start = time.time()
on_epoch_end(opts.numEpochs)
time_end = time.time()
time_inference = time_end - time_start

batchesPerEpoch = examplesPerEpoch / opts.batch_size;
totalBatches = opts.numEpochs * batchesPerEpoch;
print('------------------------------------------------')
print('trainingTimePerBach', time_training / totalBatches)
print('inferenceTimePerChar', time_inference / opts.generateLength)



