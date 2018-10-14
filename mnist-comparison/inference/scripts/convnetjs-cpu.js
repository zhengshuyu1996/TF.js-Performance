
/*
author: David Xiang
email: xdw@pku.edu.cn
 */
'use strict'
let net;
const TASK = "Inference\tconvnetjs\tcpu\t";

function initModel(){
    // later, to recreate the network:
    let json = JSON.parse(savedModel); // creates json object out of a string
    net = new convnetjs.Net(); // create an empty network
    net.fromJSON(json); // load all parameters from JSON

    // warm up the net
    for (let i = 0; i < 10; i++){
        let x = new convnetjs.Vol(1, 1, INPUT_NODE);
        net.forward(x);
    }
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

    let totTime = 0;

    let batch = await data.nextTestBatch(TEST_SIZE);
    let xs = batch.xs;  
    for (let i = 0; i < batch.labels.length/OUTPUT_NODE; i++){
        if (VERBOSE)
            console.log("Case " + i);

        let begin = new Date();

        let x = new convnetjs.Vol(1, 1, INPUT_NODE);
        for (let j = 0; j < INPUT_NODE; j++){
            x.set(0, 0, j, xs[i*INPUT_NODE+j]);
        }
        net.forward(x);
        
        let end = new Date();
        totTime += end - begin;
    }

    triggerEnd(TASK + totTime + "ms\t");
}

async function init(){
    initModel();

    let data = new MnistData();
    await data.load();

    statusLog("Ready");
    return data;
}

async function main(){
    let data = await init();
    await infer(data);
    statusLog("Finished");
}
main();

