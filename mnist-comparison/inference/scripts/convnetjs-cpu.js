
/*
author: David Xiang
email: xdw@pku.edu.cn
 */
'use strict'
var net;

function initNet(){
    // later, to recreate the network:
    let json = JSON.parse(savedModel); // creates json object out of a string
    net = new convnetjs.Net(); // create an empty network
    net.fromJSON(json); // load all parameters from JSON
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
    statusLog("Inferring");

    console.time("inference");

    let batch = await data.nextTestBatch(TEST_SIZE);
    let xs = batch.xs;  
    for (let i = 0; i < batch.labels.length/OUTPUT_NODE; i++){
        let x = new convnetjs.Vol(1, 1, INPUT_NODE);
        for (let j = 0; j < INPUT_NODE; j++){
            x.set(0, 0, j, xs[i*INPUT_NODE+j]);
        }
        net.forward(x);
        //console.log(i);
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
    initNet();
    let data = await load();
    await infer(data);
    statusLog("Finished");
}
main();

