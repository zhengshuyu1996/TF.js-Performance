'use strict'
/*
author: David Xiang
email: xdw@pku.edu.cn
 */
let model, trainer;
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
        model = new convnetjs.Net(); // create an empty network
        model.fromJSON(json); // load all parameters from JSON

        let end = new Date();
        loadTime += end - start;
    }
    triggerEnd(task + loadTime/round + "\t" + warmupTime + "\t" + inferTime/round);
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
        statusLog("Finished");            
    });
}

main();


