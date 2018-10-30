'use strict'
/*
author: David Xiang
email: xdw@pku.edu.cn
 */
let model;
let loadTime = 0;
let inferTime = 0;
let warmupTime = "cpu";
let round = 10;

async function initModel(savedModel){
    await triggerStart();
    if (verbose){
        console.log("init model");
    }

    for (let i = 0; i < round; i++){
        // later, to recreate the network:
        let json = JSON.parse(savedModel); // creates json object out of a string
        
        let start = new Date();

        // constuct the net
        model = new brain.NeuralNetwork();
        model.fromJSON(json);

        let end = new Date();
        loadTime += end - start;
    }
    triggerEnd(task + loadTime/round + "\t" + warmupTime + "\t" + inferTime/round);
}

function getStdInput(xs, labels){
    /* according to brain.js, input should be in json format:
     * [{input: [0, 0], output: [0]},
     *  {input: [1, 0], output: [1]}]
     *  input should be an array or a hash of numbers from 0 to 1
     */

    let data = [];
    for (let i = 0; i < labels.length/OUTPUT_NODE; i++){
        data.push({
            input: xs.slice(i * INPUT_NODE, (i + 1) * INPUT_NODE),
            output: labels.slice(i * OUTPUT_NODE, (i+1) * OUTPUT_NODE)
        });
    }
    return data;
}

async function init(model){
    await initModel(model);

    let data = new MnistData();
    await data.load();
    
    statusLog("Ready");
    return data;
}

function readTextFile(file, callback) {
    var rawFile = new XMLHttpRequest();
    rawFile.overrideMimeType("application/json");
    rawFile.open("GET", file, true);
    rawFile.onreadystatechange = function() {
        if (rawFile.readyState === 4 && rawFile.status == "200") {
            callback(rawFile.responseText);
        }
    }
    rawFile.send(null);
}

async function main(model){
    let argsStatus = parseArgs(); // defined in params.js
    if (argsStatus == false)
        return;

    let modelPath = "/model/brainjs/mnist-" 
        + hiddenLayerNum + "-" + hiddenLayerSize + ".json";


    readTextFile(modelPath, async function(model){
        let data = await init(model);
        statusLog("Finished");            
    });
}

main();


