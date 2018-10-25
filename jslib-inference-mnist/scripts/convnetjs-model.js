'use strict'
/*
author: David Xiang
email: xdw@pku.edu.cn
 */
let model, trainer;
let loadTime = 0;
let inferTime = 0;
let warmupTime = "cpu";

async function initModel(savedModel){
    if (verbose){
        console.log("init model");
    }

    // later, to recreate the network:
    let json = JSON.parse(savedModel); // creates json object out of a string
    
    let start = new Date();

    // constuct the net
    model = new convnetjs.Net(); // create an empty network
    model.fromJSON(json); // load all parameters from JSON

    let end = new Date();
    loadTime = end - start;
}

function getLabel(LabelOneHot){
    let num = LabelOneHot.length/OUTPUT_NODE;
    let labels = new Uint8Array(num);

    for (let i = 0; i < num; i++){
        labels[i] = 0;
        for (let j = 0; j < OUTPUT_NODE; j++){
            if (LabelOneHot[i * OUTPUT_NODE + j] == 1)
                labels[i] = j;
        }
    }
    return labels;
}

async function infer(data){
    await triggerStart();
    statusLog("Inferring");

    let batch = await data.nextTestBatch(inferSize);
    let xs = batch.xs;  

    for (let i = 0; i < batch.labels.length/OUTPUT_NODE; i++){
        if (verbose)
            console.log("Case " + i);

        let begin = new Date();

        let x = new convnetjs.Vol(1, 1, INPUT_NODE);
        for (let j = 0; j < INPUT_NODE; j++){
            x.set(0, 0, j, xs[i*INPUT_NODE+j]);
        }
        model.forward(x);
        
        let end = new Date();
        inferTime += end - begin;
    }

    triggerEnd(task + loadTime + "\t" + warmupTime + "\t" + inferTime);
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

    let modelPath = "/model/convnetjs/mnist-" 
        + hiddenLayerNum + "-" + hiddenLayerSize + ".json";


    readTextFile(modelPath, async function(model){
        let data = await init(model);
        await infer(data);
        statusLog("Finished");            
    });
}

main();


