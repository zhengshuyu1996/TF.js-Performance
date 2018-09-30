/*
author: David Xiang
email: xdw@pku.edu.cn
 */
'use strict'
const BATCH_SIZE = 64;
const TRAIN_SIZE = 60000; // one epoch
const TRAIN_BATCHES = TRAIN_SIZE / BATCH_SIZE;
const TEST_SIZE = 5000;

const IMAGE_LENGTH = 28;
const INPUT_NODE = 784;
const HIDDEN_SIZE = 512;
const OUTPUT_NODE = 10;
const NUM_CHANNELS = 1;

const LEARNING_RATE = 0.15;
var net, trainer;

function initNet(){
    // constuct the net
    net = new convnetjs.Net();
    
    let layer_defs = [];
    layer_defs.push({
        type:"input", 
        out_sx:1, 
        out_sy:1, 
        out_depth:INPUT_NODE});
    layer_defs.push({
        type:"fc", 
        num_neurons:HIDDEN_SIZE, 
        activation:"relu"});
    layer_defs.push({
        type:"fc", 
        num_neurons:HIDDEN_SIZE, 
        activation:"relu"});
    layer_defs.push({
        type:"softmax", 
        num_classes:OUTPUT_NODE});

    net.makeLayers(layer_defs);
    
    // initiate
    trainer = new convnetjs.Trainer(net, {
        learning_rate:LEARNING_RATE,
        method: "sgd",
        batch_size:BATCH_SIZE
    })
}


async function train(data){
    statusLog("Training");

    console.time("train");

    for (let i = 0; i < TRAIN_BATCHES; i++){
        let batch = data.nextTrainBatch(BATCH_SIZE);
        let xs = batch[0];
        let labels = batch[1];
        
        for (let j = 0; j < BATCH_SIZE; j++){
            let x = new convnetjs.Vol(1, 1, OUTPUT_NODE);
            let y = labels[j];
            for (let k = 0; k < OUTPUT_NODE; k++){
                x.set(1, 1, k, xs[j*OUTPUT_NODE+k]/255.0-0.5);
            }
            let stats = trainer.train(x, y);

            if (j == 0){
                let loss = stats.loss;
                console.log('Batch #' + i + "    Loss: " + loss.toFixed(3));
            }
        }
    }
    console.timeEnd("train");

    statusLog("Testing");
    let testData = data.nextTestBatch(TEST_SIZE);
    let xs = testData[0];
    let labels = testData[1];
    let count = 0;
    for (let j = 0; j < TEST_SIZE; j++){
        let x = new convnetjs.Vol(1, 1, OUTPUT_NODE);
        let y = labels[j];
        for (let k = 0; k < OUTPUT_NODE; k++){
            x.set(1, 1, k, xs[j*OUTPUT_NODE+k]/255.0-0.5);
        }
        net.forward(x);
        let y_ = net.getPredication();
        if (y_ === y)
            count++;
    }
    let acc = count / TEST_SIZE;
    console.log('accuracy: ' + acc.toFixed(3));
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
    await train(data);
    statusLog("Finished");
}
main();

