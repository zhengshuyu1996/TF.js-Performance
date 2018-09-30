/*
author: David Xiang
email: xdw@pku.edu.cn
 */
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

async function train(data){
    const model = tf.sequential();
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

    statusLog("Training");

    console.time("train");

    for (let i = 0; i < TRAIN_BATCHES; i++){
        let batch = data.nextTrainBatch(BATCH_SIZE);
        let history = await model.fit(
            batch.xs, 
            batch.labels,
            {batchSize: BATCH_SIZE, epochs: 1}
        );

        let loss = history.history.loss[0];
        console.log('Batch #' + i + "    Loss: " + loss.toFixed(3));

        tf.dispose(batch);
        await tf.nextFrame();
    }
    console.timeEnd("train");

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

async function load(){
    let data = new MnistData();
    await data.load();
    statusLog("Ready");
    return data;
}

async function init(data){
    tf.setBackend("cpu");
    console.log(tf.getBackend());
    await train(data);
    statusLog("Finished");
}
async function main(){
    let data = await load();
    await init(data);
}
main();

