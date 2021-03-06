'use strict'
/*
author: David Xiang
email: xdw@pku.edu.cn
 */
let model;

function initModel(){
    // constuct the net
    let hiddenLayers = [];
    
    for (let i = 0; i < hiddenLayerNum; i++){
        hiddenLayers.push(hiddenLayerSize);
    }

    model = new brain.NeuralNetwork({
        hiddenLayers: hiddenLayers,
        // if assign relu to activation, then the training process 
        // can not converge. Why?
        // activation: "relu"
    });
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

async function train(data){
    await triggerStart();
    statusLog("Training");

    let totTime = 0;
    let round = 0;

    while (totTime < trainTime){
        let batch = await data.nextTrainBatch(BATCH_SIZE);
        let trainData = getStdInput(batch.xs, batch.labels);

        if(verbose)
            console.log(round);

        let begin = new Date();

        model.train(trainData, {
            iterations: 1,
            learningRate: LEARNING_RATE,
            momentum: 0.000001, // if we set 0, the library will generate warning
            log: verbose, // false => only time are printed in console
            logPeriod: 1
        });
        
        let end = new Date();
        totTime += end - begin;
        round++;
    }

    triggerEnd(task + totTime/round);

    if(dotest){
        // test procedure
        statusLog("Testing");
        let batch = await data.nextTestBatch(TEST_SIZE);
        let testData = getStdInput(batch.xs, batch.labels);

        let labels = OneHot2Label(batch.labels);
        let count = 0;
        for (let j = 0; j < TEST_SIZE; j++){
            let y = labels[j]; // correct label

            // reduce: get max value's key
            let output = model.run(testData[j].input);
            let max = output[0];
            let y_ = 0;
            for (let k = 0; k < OUTPUT_NODE; k++){
                if (output[k] > max){
                    max = output[k];
                    y_ = k;
                }
            }

            // count the correct prediction
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

