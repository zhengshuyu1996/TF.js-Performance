'use strict'
/*
author: David Xiang
email: xdw@pku.edu.cn
 */

let model, trainer;

function initModel(){
    // constuct the net

    if (verbose){
        console.log("init model");
    }
    
    let inputLayer = new synaptic.Layer(INPUT_NODE);
    let hiddenLayers = [];
    let outputLayer = new synaptic.Layer(OUTPUT_NODE);

    for (let i = 0; i < hiddenLayerNum; i++){
        hiddenLayers.push(new synaptic.Layer(hiddenLayerSize));
    }

    inputLayer.project(hiddenLayers[0]);
    for (let i = 0; i < hiddenLayerNum - 1; i++){
        hiddenLayers[i].project(hiddenLayers[i+1]);
    }
    hiddenLayers[hiddenLayerNum-1].project(outputLayer);
    model = new synaptic.Network({
        input: inputLayer,
        hidden: hiddenLayers,
        output: outputLayer
    })
    trainer = new synaptic.Trainer(model);
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
    await triggerStart();
    statusLog("Training");

    let totTime = 0;

    let round = 0;
    while (totTime < trainTime){
        let batch = await data.nextTrainBatch(BATCH_SIZE);
        let trainData = getStdInput(batch.xs, batch.labels);

        if (verbose)
            console.log(i);

        let begin = new Date();

        trainer.train(trainData,{
            rate: LEARNING_RATE,
            iterations: 1,
            log: 1,
            cost: synaptic.Trainer.cost.CROSS_ENTROPY
        });

        let end = new Date();
        totTime += end - begin;
        round++;
    }
    
    triggerEnd(task + totTime/round);

    if (dotest){
        statusLog("Testing");
        let batch = await data.nextTestBatch(TEST_SIZE);
        let testData = getStdInput(batch.xs, batch.labels);

        let labels = OneHot2Label(batch.labels);
        let count = 0;
        for (let j = 0; j < TEST_SIZE; j++){
            let y = labels[j]; // correct label

            let output = model.activate(testData[j].input);
            let max = output[0];
            let y_ = 0;
            for (let k = 0; k < OUTPUT_NODE; k++){
                if (output[k] > max){
                    max = output[k];
                    y_ = k;
                }
            }
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
    
    statusLog("Initializing");
    let data = await init();
    
    await train(data);
    statusLog("Finished");
}
main();
