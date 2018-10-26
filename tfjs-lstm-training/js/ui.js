
async function triggerStart(){
    await new Promise((resolve) => setTimeout(resolve, 5000));
    // wait for 5 seconds
    let event = new CustomEvent("started");
    console.log("start");
    document.dispatchEvent(event);
}

function triggerEnd(msg){
    let event = new CustomEvent("finished", {
        "detail":{
            message: msg
        }
    });
    console.log("end");
    console.log(msg);
    document.dispatchEvent(event);
}


// UI controls.
const testText = document.getElementById('test-text');
const appStatus = document.getElementById('app-status');

// Module-global instance of TextData.
let textData;

// Module-global instance of SaveableLSTMTextGenerator.
let textGenerator;

function logStatus(message) {
  appStatus.textContent = message;
}

let lossValues;
let batchCount;

/**
 * A function to call when a training process starts.
 */
function onTrainBegin() {
  lossValues = [];
  logStatus('Starting model training...');
}

/**
 * A function to call when a batch is competed during training.
 *
 * @param {number} loss Loss value of the current batch.
 * @param {number} progress Total training progress, as a number between 0
 *   and 1.
 * @param {number} examplesPerSec The training speed in the batch, in examples
 *   per second.
 */
function onTrainBatchEnd(loss, progress, examplesPerSec) {
  batchCount = lossValues.length + 1;
  lossValues.push({'batch': batchCount, 'loss': loss, 'split': 'training'});
  plotLossValues();
  logStatus(
      `Model training: ${(progress * 1e2).toFixed(1)}% complete... ` +
      `(${examplesPerSec.toFixed(0)} examples/s)`);
}

function onTrainEpochEnd(validationLoss) {
  lossValues.push(
      {'batch': batchCount, 'loss': validationLoss, 'split': 'validation'});
  plotLossValues();
}

function plotLossValues() {
  console.log(lossValues[lossValues.length-1])
}


async function fetchData() {
  let dataIdentifier = 'Nietzsche';
  await fetch('/Nietzsche.txt')
    .then (res => {
      res.text().then(data => {
        testText.value = data;
        textData = new TextData(dataIdentifier, testText.value, sampleLen, sampleStep);
        textGenerator = new LSTMTextGenerator(textData);
        createModel();
      })
    })
}


async function createModel () {
  if (textGenerator == null) {
    logStatus('createModel ERROR: Please load text data set first.');
    return;
  }

  logStatus('Creating model... Please wait.');
  for (let i = 0; i < lstmLayerSizes.length; ++i) {
    const lstmLayerSize = lstmLayerSizes[i];
    if (!(lstmLayerSize > 0)) {
      logStatus(
          `ERROR: lstmLayerSizes must be a positive integer, ` +
          `but got ${lstmLayerSize} for layer ${i + 1} ` +
          `of ${lstmLayerSizes.length}.`);
      return;
    }
  }

  await textGenerator.createModel(lstmLayerSizes);
  logStatus(
      'Done creating model. ' +
      'Now you can train the model or use it to generate text.');

  trainModel()
}


async function trainModel () {
  if (textGenerator == null) {
    logStatus('trainModel ERROR: Please load text data set first.');
    return;
  }
  
  textGenerator.compileModel(learningRate);

  await textGenerator.fitModel(
      numEpochs, examplesPerEpoch, batchSize, validationSplit);

}

