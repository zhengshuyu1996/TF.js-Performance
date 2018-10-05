/*
author: David Xiang
email: xdw@pku.edu.cn
 */
'use strict'
let model;
tf.setBackend("cpu");
async function infer(data){
    statusLog("Inferring");

    console.time("inference");
    let batch = data.nextTestBatch(TEST_SIZE);
    for (let i = 0; i < batch.labels.length/OUTPUT_NODE; i++){
        let input = tf.tensor2d(batch.xs.slice(i * INPUT_NODE, (i + 1) * INPUT_NODE), [1, INPUT_NODE]);
        model.predict(input);
    }
    console.timeEnd("inference");
}

async function load(){
    let data = new MnistData();
    await data.load();

    // load models
    model = await tf.loadModel(LOCAL_SERVER+"/model/tfjs/mymodel.json");

    // warm up the model
    for (let i = 1; i < 10; i++)
        model.predict(tf.ones([1, INPUT_NODE])).dispose();
    
    statusLog("Ready");
    return data;
}

async function init(data){
    await infer(data);
    statusLog("Finished");
}
async function main(){
    let data = await load();
    await init(data);
}
main();

