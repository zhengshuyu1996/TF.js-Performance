
// args to be extracted from url
let lstmLayerSizes;
let backend;
let sampleLen = 40;
let sampleStep = 3;
let generateLength = 200;
let temperature = 0.75;
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
    
    if (getParam(query, "temperature")) {
        temperature = parseFloat(getParam(query, "temperature"));
    }

    if (getParam(query, "generateLength")) {
        generateLength = parseInt(getParam(query, "generateLength"));
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
    task = "jslib\tinference\tTextGeneration\ttensorflowjs\t" + rnnType + "\t" + backend + "\t" 
    + getParam(query, "layersizes") + "\t";
    document.getElementById("task").innerText = task;
    return true;

}
