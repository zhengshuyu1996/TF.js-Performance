
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

const seedTextInput = document.getElementById('seed-text');
const generatedTextInput = document.getElementById('generated-text');


// Module-global instance of TextData.
let textData;

// Module-global instance of SaveableLSTMTextGenerator.
let textGenerator;

function logStatus(message) {
  appStatus.textContent = message;
}

let startTime;
let endTime;

/**
 * A function to call when text generation begins.
 *
 * @param {string} seedSentence: The seed sentence being used for text
 *   generation.
 */
async function onTextGenerationBegin() {
  generatedTextInput.value = '';
  logStatus('Generating text...');
  await triggerStart();
  startTime = new Date().getTime();
}

async function onTextGenerationEnd() {
  endTime = new Date().getTime();
  await triggerEnd(task + "inferenceTimePerChar=" + (endTime - startTime)/generateLength);
}

/**
 * A function to call each time a character is obtained during text generation.
 *
 * @param {string} char The just-generated character.
 */
async function onTextGenerationChar(char) {
  generatedTextInput.value += char;
  generatedTextInput.scrollTop = generatedTextInput.scrollHeight;
  const charCount = generatedTextInput.value.length;
  const status = `Generating text: ${charCount}/${generateLength} complete...`;
  logStatus(status);
  await tf.nextFrame();
}


function fetchData() {
  let dataIdentifier = 'Nietzsche';
  fetch('/Nietzsche.txt')
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

  generateText();
}


async function generateText() {
  try {

    if (textGenerator == null) {
      logStatus('generateText ERROR: Please load text data set first.');
      return;
    }

    let seedSentence;
    let seedSentenceIndices;
    [seedSentence, seedSentenceIndices] = textData.getRandomSlice();
    seedTextInput.value = seedSentence;


    const sentence = await textGenerator.generateText(
        seedSentenceIndices, generateLength, temperature);
    generatedTextInput.value = sentence;
    const status = 'Done generating text.';
    logStatus(status);

  } catch (err) {
    logStatus(`ERROR: Failed to generate text: ${err.message}, ${err.stack}`);
  }
}

