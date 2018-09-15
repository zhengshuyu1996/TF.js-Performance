/*
author: David Xiang
email: xdw@pku.edu.cn
 */
console.log(tf.getBackend());
//tf.setBackend("cpu");
var btn = document.getElementById("train");
var nConv = document.getElementById("nConv");
var nPool = document.getElementById("nPool");
var equation = document.getElementById("equation");
var mPerform = document.getElementById("mPerform");
var mInfo, tInfo;

const LEARNING_RATE = 0.15;
const optimizer = tf.train.sgd(LEARNING_RATE);
const BATCH_SIZE = 64;
const TRAIN_BATCHES = 150;
const TEST_BATCH_SIZE = 1000;
const TEST_ITERATION_FREQUENCY = 5;

async function train(nconv, npool){
    const model = tf.sequential();

    for (let i = 0; i < npool; i++){
        for (let j = 0; j < nconv; j++){
            if (i == 0 && j == 0){
                model.add(tf.layers.conv2d({
                    inputShape: [28, 28, 1],
                    kernelSize: 5,
                    filters: 8,
                    strides: 1,
                    activation: 'relu',
                    kernelInitializer: 'varianceScaling',
                    padding: 'same'
                }));
            }else{
                model.add(tf.layers.conv2d({
                    kernelSize: 5,
                    filters: 8,
                    strides: 1,
                    activation: 'relu',
                    kernelInitializer: 'varianceScaling',
                    padding: 'same'
                }));
            }
        }
        model.add(tf.layers.maxPooling2d({
            poolSize: [2, 2], 
            strides: [2, 2],
            padding: 'same'
        }));
    }

    model.add(tf.layers.flatten());

    model.add(tf.layers.dense({
        units: 10,
        kernelInitializer: "varianceScaling",
        activation: "softmax"
    }));

    model.compile({
        optimizer: optimizer,
        loss: "categoricalCrossentropy",
        metrics: ['accuracy']
    });

    statusLog("Training");

    for (let i = 0; i < TRAIN_BATCHES; i++){
        const [batch, validationData] = tf.tidy(()=>{
            const batch = data.nextTrainBatch(BATCH_SIZE);
            batch.xs = batch.xs.reshape([BATCH_SIZE, 28, 28, 1]);

            let validationData;
            if (i % TEST_ITERATION_FREQUENCY === 0){
                const testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
                validationData = [
                    testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]), testBatch.labels
                ];
            }
            return [batch, validationData];
        });

        let history;
        //let info = await tf.time(async()=>{
            history = await model.fit(
                batch.xs, 
                batch.labels,
                {batchSize: BATCH_SIZE, validationData, epochs: 1});
        //});

        const loss = history.history.loss[0];
        const accuracy = history.history.acc[0];

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
    //tInfo = await tf.time(()=>{
        await train(nconv, npool);
    //});
    console.timeEnd("train");

    mPerform.innerText = "memory performance: " + JSON.stringify(mInfo);
    tPerform.innerText = "time performance: " + JSON.stringify(tInfo);
    statusLog("Finished");
    console.log(tf.getBackend());
};

load();
console.log("executed.");

