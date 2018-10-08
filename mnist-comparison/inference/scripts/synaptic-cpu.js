/*
author: David Xiang
email: xdw@pku.edu.cn
 */
'use strict'
var net;

function initNet(){
    let json = JSON.parse(savedModel);

    // constuct the net
    net = synaptic.Network.fromJSON(json);

    // warm up the net
    let arr = new Array(INPUT_NODE);
    let testInput = {
        input: arr
    }
    for (let i = 0; i < 10; i++){
        net.activate(testInput);
    }
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

async function train(data){
    statusLog("Inferring");

    console.time("inference");
    
    let batch = await data.nextTestBatch(TEST_SIZE);
    let TestData = getStdInput(batch.xs, batch.labels);

    let i = 0;
    for (let item in TestData){
        //console.log(i++);
        net.activate(item);
    }

    console.timeEnd("inference");
}

async function load(){
    let data = new MnistData();
    await data.load();
    statusLog("Ready");
    return data;
}

async function main(){
    statusLog("Initializing Network");
    initNet();
    statusLog("Loading");
    let data = await load();
    await train(data);
    statusLog("Finished");
}
main();
