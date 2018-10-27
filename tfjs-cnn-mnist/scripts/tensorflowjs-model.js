'use strict'
/*
author: David Xiang
email: xdw@pku.edu.cn
 */

let model;
let loadTime, warmupTime;
let avgTrainTime, avgInferTime;

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

    for (let i = 0; i < units; i++){
        if (i == 0){
            model.add(tf.layers.conv2d({
                inputShape: [IMAGE_LENGTH, IMAGE_LENGTH, 1],
                kernelSize: kernelSize,
                filters: filters,
                strides: 1,
                activation: 'relu',
                padding: 'same'
            }));
        }else{
            model.add(tf.layers.conv2d({
                kernelSize: kernelSize,
                filters: filters,        
                strides: 1,
                activation: 'relu',
                padding: 'same'
            }));
        }
        model.add(tf.layers.conv2d({
            kernelSize: kernelSize,
            filters: filters,        
            strides: 1,
            activation: 'relu',
            padding: 'same'
        }));
        model.add(tf.layers.maxPooling2d({
            poolSize: poolSize, 
            strides: poolSize,
            padding: 'same'
        }));
    }

    model.add(tf.layers.flatten());

    // output layer
    model.add(tf.layers.dense({
        units: OUTPUT_NODE,
        activation: "softmax"
    }));

    model.compile({
        optimizer: optimizer,
        loss: "categoricalCrossentropy",
        metrics: ['accuracy']
    });
    //console.log("end init model");
}

async function trainAndInfer(data){
    //console.log("start process");
    await triggerStart();

    // start training
    statusLog("Training");

    let totTime = 0;

    let round = 0;
    while (totTime < trainTime){
        let batch = await data.nextTrainBatch(batchSize);
        batch.xs = batch.xs.reshape([batchSize, IMAGE_LENGTH, IMAGE_LENGTH, 1]);

        let begin = new Date();

        let history = await model.fit(
            batch.xs, 
            batch.labels,
            {batchSize: batchSize, epochs: 1}
        );

        let loss = history.history.loss[0];
        let accuracy = history.history.acc[0];

        tf.dispose(batch);
        await tf.nextFrame();

        let end = new Date();
        totTime += end - begin;

        if (verbose){
            console.log('Batch #' + round + "    Loss: " + loss.toFixed(3) +
                "    Accuracy: " + accuracy.toFixed(3));
        }
        round++;
    }

    avgTrainTime = totTime / round;
    
    // save results
    const saveResults = await model.save('indexeddb://model');

    let start = new Date();


    statusLog("Loading Model");

    // load models
    model = await tf.loadModel('indexeddb://model');

    let end = new Date();
    loadTime = end - start;

    start = new Date();
    // warm up the model
    model.predict(tf.ones([1, IMAGE_LENGTH, IMAGE_LENGTH, 1])).dispose();
    end = new Date();
    if (backend == "gpu")
        warmupTime = end - start;
    else
        warmupTime = "cpu"
    
    // start inference
    statusLog("Inferring");

    round = 0;
    totTime = 0;
    let inputTensor = tf.ones([1, IMAGE_LENGTH, IMAGE_LENGTH, 1]);
    while(totTime < inferTime){
        inputTensor = inputTensor.add(tf.ones([1, IMAGE_LENGTH, IMAGE_LENGTH, 1]));
        
        if (verbose)
            console.log("Case " + round);

        let begin = new Date();

        model.predict(inputTensor);

        let end = new Date();
        
        totTime += end - begin;
        round++;
    }    

    avgInferTime = totTime / round;

    triggerEnd(task + avgTrainTime + "\t" +loadTime + "\t" + warmupTime + "\t" + avgInferTime);
}

async function init(){
    await initModel();

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
    await trainAndInfer(data);
    statusLog("Finished");
}
main();

