/*
author: David Xiang
email: xdw@pku.edu.cn
 */
'use strict';
const TASK = "Training\ttfjs\tgpu\t";
var model;

async function initNet(){
    model = tf.sequential();
    const optimizer = tf.train.sgd(LEARNING_RATE);

    // hidden layer 1
    model.add(tf.layers.dense({
        inputShape: [INPUT_NODE],
        units: HIDDEN_SIZE,
        activation: "relu",
    }));

    // hidden layer 2
    model.add(tf.layers.dense({
        units: HIDDEN_SIZE,
        activation: "relu",
    }));

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
    triggerStart();
    statusLog("Training");

    let totTime = 0;

    for (let i = 0; i < TRAIN_BATCHES; i++){
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

        if (VERBOSE){
            console.log('Batch #' + i + "    Loss: " + loss.toFixed(3));
        }
    }

    if (DO_TEST){
        statusLog("Testing");
        let batch = data.nextTrainBatch(BATCH_SIZE);
        let testData = data.nextTestBatch(TEST_SIZE);
        let history = await model.fit(
            batch.xs, 
            batch.labels,
            {batchSize: BATCH_SIZE, testData, epochs: 1}
        );

        let acc = history.history.acc[0];
        console.log('accuracy: ' + acc.toFixed(3));
    }

    triggerEnd(TASK + "time:\t" + totTime + "ms\t");
}

async function init(){
    tf.setBackend("webgl");
    console.log(tf.getBackend());
    await initNet();

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

