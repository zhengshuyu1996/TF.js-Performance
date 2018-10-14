/*
author: David Xiang
email: xdw@pku.edu.cn
 */
'use strict'
var net;
const TASK = "Training\tbrainjs\tcpu\t";

function initNet(){
    // constuct the net
    net = new brain.NeuralNetwork({
        hiddenLayers: [512, 512],
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
    triggerStart();
    statusLog("Training");

    let totTime = 0;

    for (let i = 0; i < TRAIN_BATCHES; i++){
        let batch = await data.nextTrainBatch(BATCH_SIZE);
        let trainData = getStdInput(batch.xs, batch.labels);

        if(VERBOSE)
            console.log(i);

        let begin = new Date();

        net.train(trainData, {
            iterations: 1,
            learningRate: LEARNING_RATE,
            momentum: 0.000001, // if we set 0, the library will generate warning
            log: VERBOSE, // false => only time are printed in console
            logPeriod: 1
        });
        
        let end = new Date();
        totTime += end - begin;
    }

    if(DO_TEST){
        // test procedure
        statusLog("Testing");
        let batch = await data.nextTestBatch(TEST_SIZE);
        let testData = getStdInput(batch.xs, batch.labels);

        let labels = OneHot2Label(batch.labels);
        let count = 0;
        for (let j = 0; j < TEST_SIZE; j++){
            let y = labels[j]; // correct label

            // reduce: get max value's key
            let output = net.run(testData[j].input);
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

