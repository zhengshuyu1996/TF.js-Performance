/*
author: David Xiang
email: xdw@pku.edu.cn
 */
'use strict'
let model;
async function initData(){

}
async function initModel(){
    //set backend
    tf.setBackend("webgl");
    console.log(tf.getBackend());

    // load models
    console.log("loading model");
    statusLog("Loading Model")
    model = await tf.loadModel(LOCAL_SERVER+"/model/tfjs/vgg19/model.json");

    // warm up the model
    console.log("warmup");
    statusLog("Warming up");
    for (let i = 1; i < 10; i++)
        model.predict(tf.ones([1, 224, 224, 3])).dispose();
}

async function infer(){
    statusLog("Inferring");

    console.time("inference");
    let batch = data.nextTestBatch(TEST_SIZE);
    for (let i = 0; i < batch.labels.length/OUTPUT_NODE; i++){
        let input = tf.tensor2d(batch.xs.slice(i * INPUT_NODE, (i + 1) * INPUT_NODE), [1, INPUT_NODE]);
        model.predict(input);
    }
    console.timeEnd("inference");
}

async function init(){
    console.log("Ready");
}


async function main(){
    let data = await load();
    await init(data);
}
main();

