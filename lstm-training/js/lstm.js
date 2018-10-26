/**
 * TensorFlow.js Example: LSTM Text Generation.
 *
 * Inspiration comes from:
 *
 * -
 * https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
 * - Andrej Karpathy. "The Unreasonable Effectiveness of Recurrent Neural
 * Networks" http://karpathy.github.io/2015/05/21/rnn-effectiveness/
 */


/**
 * Class that manages LSTM-based text generation.
 *
 * This class manages the following:
 *
 * - Creating and training a LSTM model, written with the tf.layers API, to
 *   predict the next character given a sequence of input characters.
 * - Generating random text using the LSTM model.
 */
class LSTMTextGenerator {
  /**
   * Constructor of NeuralNetworkTextGenerator.
   *
   * @param {TextData} textData An instance of `TextData`.
   */
  constructor(textData) {
    this.textData_ = textData;
    this.charSetSize_ = textData.charSetSize();
    this.sampleLen_ = textData.sampleLen();
    this.textLen_ = textData.textLen();
  }

  /**
   * Create LSTM model from scratch.
   *
   * @param {number | number[]} lstmLayerSizes Sizes of the LSTM layers, as a
   *   number or an non-empty array of numbers.
   */
  createModel(lstmLayerSizes) {
    if (!Array.isArray(lstmLayerSizes)) {
      lstmLayerSizes = [lstmLayerSizes];
    }

    if (backend == "cpu")
        tf.setBackend("cpu");
    else
        tf.setBackend("webgl");
    if (verbose){
        console.log(tf.getBackend());
        console.log("init model");
    }

    this.model = tf.sequential();
    for (let i = 0; i < lstmLayerSizes.length; ++i) {
      const lstmLayerSize = lstmLayerSizes[i];
      this.model.add(tf.layers.lstm({
        units: lstmLayerSize,
        returnSequences: i < lstmLayerSizes.length - 1,
        inputShape: i === 0 ? [this.sampleLen_, this.charSetSize_] : undefined
      }));
    }
    this.model.add(
        tf.layers.dense({units: this.charSetSize_, activation: 'softmax'}));
  }

  /**
   * Compile model for training.
   *
   * @param {number} learningRate The learning rate to use during training.
   */
  compileModel(learningRate) {
    const optimizer = tf.train.rmsprop(learningRate);
    this.model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
    console.log(`Compiled model with learning rate ${learningRate}`);
    this.model.summary();
  }

  /**
   * Train the LSTM model.
   *
   * @param {number} numEpochs Number of epochs to train the model for.
   * @param {number} examplesPerEpoch Number of epochs to use in each training
   *   epochs.
   * @param {number} batchSize Batch size to use during training.
   * @param {number} validationSplit Validation split to be used during the
   *   training epochs.
   */
  async fitModel(numEpochs, examplesPerEpoch, batchSize, validationSplit) {
    let batchCount = 0;
    const batchesPerEpoch = examplesPerEpoch / batchSize;
    const totalBatches = numEpochs * batchesPerEpoch;

    onTrainBegin();
    await tf.nextFrame();

    await triggerStart();
    let t = new Date().getTime();
    let start = new Date().getTime();

    for (let i = 0; i < numEpochs; ++i) {
      const [xs, ys] = this.textData_.nextDataEpoch(examplesPerEpoch);
      await this.model.fit(xs, ys, {
        epochs: 1,
        batchSize: batchSize,
        // validationSplit,
        callbacks: {
          onBatchEnd: async (batch, logs) => {
            // Calculate the training speed in the current batch, in # of
            // examples per second.
            const t1 = new Date().getTime();
            const examplesPerSec = batchSize / ((t1 - t) / 1e3);
            onTrainBatchEnd(
                logs.loss, ++batchCount / totalBatches, examplesPerSec);
            if (t1 - start >= timeLimit) {
              await triggerEnd(task + (t1 - start)/batchCount);
            }
            t = t1;
          },
          onEpochEnd: async (epoch, logs) => {
            onTrainEpochEnd(logs.val_loss);
          },
        }
      });
      xs.dispose();
      ys.dispose();
    }

    // triggerEnd(task + t);
  }

  /**
   * Generate text using the LSTM model.
   *
   * @param {number[]} sentenceIndices Seed sentence, represented as the
   *   indices of the constituent characters.
   * @param {number} length Length of the text to generate, in number of
   *   characters.
   * @param {number} temperature Temperature parameter. Must be a number > 0.
   * @returns {string} The generated text.
   */
  async generateText(sentenceIndices, length, temperature) {
    onTextGenerationBegin();
    const temperatureScalar = tf.scalar(temperature);

    let generated = '';
    while (generated.length < length) {
      // Encode the current input sequence as a one-hot Tensor.
      const inputBuffer =
          new tf.TensorBuffer([1, this.sampleLen_, this.charSetSize_]);
      for (let i = 0; i < this.sampleLen_; ++i) {
        inputBuffer.set(1, 0, i, sentenceIndices[i]);
      }
      const input = inputBuffer.toTensor();

      // Call model.predict() to get the probability values of the next
      // character.
      const output = this.model.predict(input);

      // Sample randomly based on the probability values.
      const winnerIndex = sample(tf.squeeze(output), temperatureScalar);
      const winnerChar = this.textData_.getFromCharSet(winnerIndex);
      await onTextGenerationChar(winnerChar);

      generated += winnerChar;
      sentenceIndices = sentenceIndices.slice(1);
      sentenceIndices.push(winnerIndex);

      input.dispose();
      output.dispose();
    }
    temperatureScalar.dispose();
    return generated;
  }
};


async function main () {
  let argsStatus = parseArgs(); // defined in params.js
  if (argsStatus == false)
      return;

  // let model = await fetchData();
  // model = await createModel(model);
  // await trainModel(model);

  fetchData();
}

main();



