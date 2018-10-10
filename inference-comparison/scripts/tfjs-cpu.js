/*
author: David Xiang
email: xdw@pku.edu.cn
 */
'use strict'
let model;
tf.setBackend("cpu");
/*async function infer(){
    statusLog("Inferring");

    console.time("inference");
    let batch = data.nextTestBatch(TEST_SIZE);
    for (let i = 0; i < batch.labels.length/OUTPUT_NODE; i++){
        let input = tf.tensor2d(batch.xs.slice(i * INPUT_NODE, (i + 1) * INPUT_NODE), [1, INPUT_NODE]);
        model.predict(input);
    }
    console.timeEnd("inference");
}*/

async function load(){
    // load testData
    // load models
    console.log("load");
    model = await tf.loadModel(LOCAL_SERVER+"/model/tfjs/vgg19/model.json");

    console.log("warmup");
    // warm up the model
    for (let i = 1; i < 10; i++)
        model.predict(tf.ones([1, 224, 224, 3])).dispose();
    
    console.log("Ready");
    return data;
}

async function init(){
    //await infer(data);
    console.log("Finished");
}

async function main(){
    console.log(tf.getBackend());
    let data = await load();
    await init(data);
}
main();

