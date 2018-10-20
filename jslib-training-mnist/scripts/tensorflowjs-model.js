'use strict'
/*
author: David Xiang
email: xdw@pku.edu.cn
 */

let model;

async function initModel(){
    //set backend
    if (backend == "cpu")
        tf.setBackend("cpu");
    else
        tf.setBackend("webgl");

    // load model
    if (verbose){
        console.log(tf.getBackend());
        console.log("init model");
    }

    model = tf.sequential();
    const optimizer = tf.train.sgd(LEARNING_RATE);


    for (let i = 0; i < hiddenLayer; i++){
        if(i == 0){
            model.add(tf.layers.dense({
                inputShape: [INPUT_NODE],
                units: hiddenSize,
                activation: "relu",
            }));
        }else{
            model.add(tf.layers.dense({
                units: hiddenSize,
                activation: "relu",
            }));
        }
    }

    // output layer
    model.add(tf.layers.dense({
        units: OUTPUT_NODE,
        kernelInitializer: tf.initializers.truncatedNormal({mean:0, stddev:0.1}),
        activation: "softmax"
    }));

    model.compile({
        optimizer: optimizer,
        loss: "categoricalCrossentropy",
        metrics: ['accuracy']
    });

}

async function train(data){
    await triggerStart();
    statusLog("Training");

    let totTime = 0;

    for (let i = 0; i < trainBatch; i++){
        let batch = data.nextTrainBatch(BATCH_SIZE);

        let begin = new Date();

        let history = await model.fit(
            batch.xs, 
            batch.labels,
            {batchSize: BATCH_SIZE, epochs: 1}
        );

        let loss = history.history.loss[0];

        tf.dispose(batch);
        await tf.nextFrame();

        let end = new Date();
        totTime += end - begin;

        if (verbose){
            console.log('Batch #' + i + "    Loss: " + loss.toFixed(3));
        }
    }

    triggerEnd(task + totTime);
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

