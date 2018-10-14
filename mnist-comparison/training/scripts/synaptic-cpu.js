/*
author: David Xiang
email: xdw@pku.edu.cn
 */
'use strict'
var net, trainer;
const TASK = "Training\tsynaptic\tcpu\t"

function initNet(){
    // constuct the net
    let inputLayer = new synaptic.Layer(INPUT_NODE);
    let hiddenLayer1 = new synaptic.Layer(HIDDEN_SIZE);
    let hiddenLayer2 = new synaptic.Layer(HIDDEN_SIZE);
    let outputLayer = new synaptic.Layer(OUTPUT_NODE);
    inputLayer.project(hiddenLayer1);
    hiddenLayer1.project(hiddenLayer2);
    hiddenLayer2.project(outputLayer);
    net = new synaptic.Network({
        input: inputLayer,
        hidden: [hiddenLayer1, hiddenLayer2],
        output: outputLayer
    })
    trainer = new synaptic.Trainer(net);
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

    for (let i = 0; i < TRAIN_BATCHES; i++){
        let batch = await data.nextTrainBatch(BATCH_SIZE);
        let trainData = getStdInput(batch.xs, batch.labels);

        if (VERBOSE)
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
    }

    if (DO_TEST){
        statusLog("Testing");
        let batch = await data.nextTestBatch(TEST_SIZE);
        let testData = getStdInput(batch.xs, batch.labels);

        let labels = OneHot2Label(batch.labels);
        let count = 0;
        for (let j = 0; j < TEST_SIZE; j++){
            let y = labels[j]; // correct label

            let output = net.activate(testData[j].input);
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
    triggerEnd(TASK + totTime + "ms\t");
}

async function init(){
    initNet();

    let data = new MnistData();
    await data.load();

    statusLog("Ready");
    return data;
}

async function main(){
    statusLog("Initializing");
    let data = await init();
    
    await train(data);
    statusLog("Finished");
}
main();
