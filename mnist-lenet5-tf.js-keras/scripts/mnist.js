/*
author: David Xiang
email: xdw@pku.edu.cn
 */
let btn = document.getElementById("train");
let nConv = document.getElementById("nConv");
let nPool = document.getElementById("nPool");
let equation = document.getElementById("equation");

const LEARNING_RATE = 0.15;
const optimizer = tf.train.sgd(LEARNING_RATE);
const BATCH_SIZE = 64;
const TRAIN_BATCHES = 1500;
const TEST_BATCH_SIZE = 1000;
const TEST_ITERATION_FREQUENCY = 5;
const IMAGE_LENGTH = 28;
const INPUT_NODE = 784;
const OUTPUT_NODE = 10;
const NUM_CHANNELS = 1;

const CONV1_SIZE = 5;
const CONV1_DEEP = 6;
const CONV2_SIZE = 5;
const CONV2_DEEP = 16;
const FLATTEN = 784;
const DENSE1_SIZE = 120;
const DENSE2_SIZE = 84;

async function train(nconv, npool){
    const model = tf.sequential();

    // conv1 
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_LENGTH, IMAGE_LENGTH, NUM_CHANNELS],
        kernelSize: CONV1_SIZE,
        filters: CONV1_DEEP,
        strides: 1,
        activation: 'relu',
        kernelInitializer: tf.initializers.truncatedNormal({mean=0, stddev:0.1}),
        padding: 'same',
        //useBias: true,
        //biasInitializer: tf.initializers.constant({value:0.0})
    }));

    // pool1
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2],
        padding: 'same'
    }));

    // conv2
    model.add(tf.layers.conv2d({
        kernelSize: CONV2_SIZE,
        filters: CONV2_DEEP,
        strides: 1,
        activation: 'relu',
        kernelInitializer: tf.initializers.truncatedNormal({mean=0, stddev:0.1}),
        padding: 'valid',
        //useBias: true,
        //biasInitializer: tf.initializers.constant({value:0.0})
    }));

    // pool2
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2],
        padding: 'same'
    }));

    // flatten
    model.add(tf.layers.flatten());

    // dense1
    model.add(tf.layers.dense({
        units: DENSE1_SIZE,
        activation: "relu",
        useBias: true,
        biasInitializer: tf.initializers.constant({value:0.0}),
        kernelInitializer: tf.initializers.truncatedNormal({mean=0, stddev:0.1}),
        //kernelRegularizer: tf.regularizers.l2()
    }));

    // dense2
    model.add(tf.layers.dense({
        units: DENSE2_SIZE,
        activation: "relu",
        useBias: true,
        biasInitializer: tf.initializers.constant({value:0.0}),
        kernelInitializer: tf.initializers.truncatedNormal({mean=0, stddev:0.1}),
        //kernelRegularizer: tf.regularizers.l2()
    }));

    // dense3
    model.add(tf.layers.dense({
        units: OUTPUT_NODE,
        kernelInitializer: tf.initializers.truncatedNormal({mean=0, stddev:0.1}),
        activation: "softmax"
    }));

    model.compile({
        optimizer: optimizer,
        loss: "categoricalCrossentropy",
        metrics: ['accuracy']
    });

    statusLog("Training");

    for (let i = 0; i < TRAIN_BATCHES; i++){
        let [batch, validationData] = tf.tidy(()=>{
            let batch = data.nextTrainBatch(BATCH_SIZE);
            batch.xs = batch.xs.reshape(
                [BATCH_SIZE, IMAGE_LENGTH, IMAGE_LENGTH, NUM_CHANNELS]);

            let validationData;
            if (i % TEST_ITERATION_FREQUENCY === 0){
                let testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
                validationData = [
                    testBatch.xs.reshape([
                        TEST_BATCH_SIZE, IMAGE_LENGTH, IMAGE_LENGTH, NUM_CHANNELS]), 
                    testBatch.labels
                ];
            }
            return [batch, validationData];
        });

        let history = await model.fit(
            batch.xs, 
            batch.labels,
            {batchSize: BATCH_SIZE, validationData, epochs: 1}
        );

        let loss = history.history.loss[0];
        let accuracy = history.history.acc[0];

        if (validationData != null)
            infoLog('Batch #' + i + "    Loss: " + loss.toFixed(3) + 
                "    Accuracy: " + accuracy.toFixed(3));
        else
            infoLog('Batch #' + i + "    Loss: " + loss.toFixed(3));

        // get memory performance
        if (i == parseInt(TRAIN_BATCHES / 2)){
            mInfo = tf.memory();
            //console.log(JSON.stringify(mInfo));
        }

        tf.dispose([batch, validationData]);
        //let info2 = await tf.time(async()=>{
            await tf.nextFrame();
            //});
        //console.log(JSON.stringify(info));
    }
}


let data;
async function load(){
    data = new MnistData();
    await data.load();
    statusLog("Ready");
}

btn.onclick = async function(){
    nconv = parseInt(nConv.value);
    npool = parseInt(nPool.value);
    equation.innerText = "total num of layers: (" + nconv + " + 1) * "
        + npool + " + 2 = " + ((nconv + 1) * npool + 2);
    
    console.log("start training");
    console.time("train");
    await train(nconv, npool);
    console.timeEnd("train");

    mPerform.innerText = "memory performance: " + JSON.stringify(mInfo);
    statusLog("Finished");
    console.log(tf.getBackend());
};

load();

