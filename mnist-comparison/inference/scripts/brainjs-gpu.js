/*
author: David Xiang
email: xdw@pku.edu.cn
 */
'use strict'
let net;
const TASK = "Inferrence\tbrainjs\tgpu\t";

async function initModel(){
    let json = JSON.parse(savedModel);

    // constuct the net
    net = new brain.NeuralNetworkGPU();
    net.fromJSON(json);

    // warm up the net
    let arr = new Array(INPUT_NODE);
    let testInput = {
        input: arr
    }
    for (let i = 0; i < 10; i++){
        net.run(testInput);
    }
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

async function infer(data){
    await triggerStart();
    statusLog("Inferring");

    let totTime = 0;
    
    let batch = await data.nextTestBatch(TEST_SIZE);
    let TestData = getStdInput(batch.xs, batch.labels);

    let i = 0;
    for (let item in TestData){
        if (VERBOSE)
            console.log("Case" + i);

        let begin = new Date();

        net.run(item);

        let end = new Date();
        totTime += end - begin;
    }

    triggerEnd(TASK + totTime + "ms\t");
}

async function init(){
    await initModel();

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

