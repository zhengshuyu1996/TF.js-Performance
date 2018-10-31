
// args to be extracted from url
let lstmLayerSizes;
let backend;
let numEpochs = 1;
let examplesPerEpoch = 2048;
let batchSize = 128;
let validationSplit = 0;
let learningRate = 1e-2;
let sampleLen = 40;
let sampleStep = 3;
let timeLimit = 10000;
let rnnType = 'LSTM';

let task;

let verbose = true;
let dotest = true;

// let libList = ["tensorflowjs", "brainjs", "synaptic", "convnetjs"];
let backendList = ["cpu", "gpu"];

function getParam(query, key){
    let regex = new RegExp(key+"=([^&]*)","i");
    let result = query.match(regex);
    if (result)
        return result[1];
    else
        return null;
}


function parseArgs(){
    let address = document.location.href;
    let query = address.split("?")[1];

    backend = getParam(query, "backend");
    lstmLayerSizes = getParam(query, "layersizes").split(',').map(s => Number.parseInt(s));

    // check whether these params are valid
    if (backendList.indexOf(backend) === -1){
        triggerStart();
        triggerEnd("Invalid URI:" + address);
        console.error("Invalid URI:" + address);
        return false;
    }

    if (lstmLayerSizes.length === 0) {
        triggerStart();
        triggerEnd("Invalid URI:" + address);
        console.error("Invalid URI:" + address);
        return false;
    }

    if (getParam(query, "processtime")) {
        timeLimit = parseInt(getParam(query, "processtime"));
    }
    
    if (getParam(query, "numEpochs")) {
        numEpochs = parseInt(getParam(query, "numEpochs"));
    }

    if (getParam(query, "examplesPerEpoch")) {
        examplesPerEpoch = parseInt(getParam(query, "examplesPerEpoch"));
    }

    if (getParam(query, "batchSize")) {
        batchSize = parseInt(getParam(query, "batchSize"));
    }

    if (getParam(query, "validation")) {
        validation = parseFloat(getParam(query, "validation"));
    }

    if (getParam(query, "learningRate")) {
        learningRate = parseFloat(getParam(query, "learningRate"));
    }

    if (getParam(query, "sampleLen")) {
        sampleLen = parseInt(getParam(query, "sampleLen"));
    }

    if (getParam(query, "sampleStep")) {
        sampleStep = parseInt(getParam(query, "sampleStep"));
    }

    if (getParam(query, "rnnType")) {
        rnnType = getParam(query, "rnnType");
    }

    // get right task name
    task = "jslib\ttraining\tTextGeneration\ttensorflowjs\t" + rnnType + "\t" + backend + "\t" 
    + getParam(query, "layersizes") + "\t" + examplesPerEpoch + "\t" + batchSize
    + "\t";
    document.getElementById("task").innerText = task;
    return true;

}
