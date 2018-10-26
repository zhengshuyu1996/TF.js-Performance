'use strict'
/*
author: David Xiang
email: xdw@pku.edu.cn
 */
let model, trainer;

function initModel(){
    // constuct the net
    model = new convnetjs.Net();
    
    let layer_defs = [];
    layer_defs.push({
        type:"input", 
        out_sx:1, 
        out_sy:1, 
        out_depth:INPUT_NODE});

    for (let i = 0; i < hiddenLayerNum; i++){
        layer_defs.push({
            type:"fc", 
            num_neurons:hiddenLayerSize, 
            activation:"relu"});
    }
    
    layer_defs.push({
        type:"softmax", 
        num_classes:OUTPUT_NODE});

    model.makeLayers(layer_defs);
    
    // initiate
    trainer = new convnetjs.Trainer(model, {
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
    await triggerStart();
    statusLog("Training");

    let totTime = 0;

    let round = 0;
    while (totTime < trainTime){
        let batch = await data.nextTrainBatch(BATCH_SIZE);
        let xs = batch.xs;
        let labelsOneHot = batch.labels;
        let labels = getLabel(labelsOneHot);
        
        if (verbose)
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
        round++;
    }

    triggerEnd(task + totTime/round);

    if (dotest){
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
            model.forward(x);
            let y_ = model.getPrediction();
            if (y_ === y){
                count++;
            }
        }
        let acc = count / TEST_SIZE;
        console.log('accuracy: ' + acc.toFixed(3));
    }

}

async function init(){
    initModel();
    
    let data = new MnistData();
    await data.load();
    
    statusLog("Ready");
    return data;
}

async function main(){
    let argsStatus = parseArgs(); // defined in params.js
    if (argsStatus == false)
        return;

    let data = await init();
    await train(data);
    statusLog("Finished");
}
main();

