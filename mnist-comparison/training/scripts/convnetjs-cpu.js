
/*
author: David Xiang
email: xdw@pku.edu.cn
 */
'use strict'
var net, trainer;
const TASK = "Training\tconvnetjs\tcpu\t";

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
        batch_size:BATCH_SIZE,
        momentum: 0.0
    })
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
async function train(data){
    triggerStart();
    statusLog("Training");

    let totTime = 0;

    for (let i = 0; i < TRAIN_BATCHES; i++){
        let batch = await data.nextTrainBatch(BATCH_SIZE);
        let xs = batch.xs;
        let labelsOneHot = batch.labels;
        let labels = getLabel(labelsOneHot);
        
        if (VERBOSE)
                console.log(i)
        
        for (let j = 0; j < BATCH_SIZE; j++){
            let x = new convnetjs.Vol(1, 1, INPUT_NODE);
            let y = labels[j];
            for (let k = 0; k < INPUT_NODE; k++){
                x.set(0, 0, k, xs[j*INPUT_NODE+k]);
            }

            let begin = new Date();

            let stats = trainer.train(x, y);

            let end = new Date();
            totTime += end - begin;
            //loss += stats.loss;
            //loss value is incompatible with tfjs?
            //if (j == BATCH_SIZE - 1){
                //console.log('Batch #' + i + "    Loss: " + (loss/BATCH_SIZE).toFixed(3));
            //}
        }
    }

    if (DO_TEST){
        statusLog("Testing");
        let testData = await data.nextTestBatch(TEST_SIZE);
        let xs = testData.xs;
        let testlabelsOneHot = testData.labels;
        let testlabels = getLabel(testlabelsOneHot);
        let count = 0;
        for (let j = 0; j < TEST_SIZE; j++){
            let x = new convnetjs.Vol(1, 1, INPUT_NODE);
            let y = testlabels[j];
            for (let k = 0; k < INPUT_NODE; k++){
                x.set(0, 0, k, xs[j*INPUT_NODE+k]);
            }
            net.forward(x);
            let y_ = net.getPrediction();
            if (y_ === y){
                count++;
            }
        }
        let acc = count / TEST_SIZE;
        console.log('accuracy: ' + acc.toFixed(3));
    }

    triggerEnd(TASK + "time:\t" + totTime + "ms\t");
}

async function init(){
    initNet();
    
    let data = new MnistData();
    await data.load();
    
    statusLog("Ready");
    return data;
}

async function main(){
    let data = await init();
    await train(data);
    statusLog("Finished");
}
main();

