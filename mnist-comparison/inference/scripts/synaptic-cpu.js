/*
author: David Xiang
email: xdw@pku.edu.cn
 */
'use strict'
var net;
const TASK = "Inference\tsynaptic\tcpu\t";

async function initModel(){
    let json = JSON.parse(savedModel);

    // constuct the net
    net = await synaptic.Network.fromJSON(json);

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

async function infer(data){
    triggerStart();
    statusLog("Inferring");

    let totTime = 0;
    
    let batch = await data.nextTestBatch(TEST_SIZE);
    let TestData = getStdInput(batch.xs, batch.labels);

    let i = 0;
    for (let item in TestData){
        console.log("Case "+i++);

        let begin = new Date();
        net.activate(item);
        let end = new Date();
        totTime += end - begin;
    }

    triggerEnd(TASK + "time:\t" + totTime + "ms\t");
}

async function init(){
    registerListener();
    await initModel();

    let data = new MnistData();
    await data.load();
    
    statusLog("Ready");
    return data;
}

async function main(){
    statusLog("Initializing");

    let data = await init();

    await infer(data);
    statusLog("Finished");
}
main();
