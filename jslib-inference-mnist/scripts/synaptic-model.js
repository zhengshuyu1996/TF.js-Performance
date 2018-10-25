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

    let json = JSON.parse(savedModel);
    
    let start = new Date();

     // constuct the net
    model = await synaptic.Network.fromJSON(json);

    let end = new Date();
    loadTime = end - start;
}

function getStdInput(xs, labels){
    let data = [];
    for (let i = 0; i < labels.length/OUTPUT_NODE; i++){
        data.push({
            input: xs.slice(i * INPUT_NODE, (i + 1) * INPUT_NODE),
            output: labels.slice(i * OUTPUT_NODE, (i+1) * OUTPUT_NODE)
        });
    }
    return data;
}

async function infer(data){
    await triggerStart();
    statusLog("Inferring");

    let batch = await data.nextTestBatch(inferSize);
    let TestData = getStdInput(batch.xs, batch.labels);

    let i = 0;
    for (let item in TestData){
        if (verbose)
            console.log("Case " + i++);

        let begin = new Date();

        model.activate(item);

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

    let modelPath = "/model/synaptic/mnist-" 
        + hiddenLayerNum + "-" + hiddenLayerSize + ".json";


    readTextFile(modelPath, async function(model){
        let data = await init(model);
        await infer(data);
        statusLog("Finished");            
    });
}

main();



