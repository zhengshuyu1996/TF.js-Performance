'use strict'
/*
author: David Xiang
email: xdw@pku.edu.cn
 */

let model;
let inferTime = 0, loadTime = 0, warmupTime = "cpu";

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

    let start = new Date();

    // load models
    let path = LOCALHOST+"/model/tensorflowjs/mnist-" + 
        + hiddenLayerNum + "-" + hiddenLayerSize + "/model.json";

    model = await tf.loadModel(path);

    let end = new Date();
    loadTime = end - start;

    
    start = new Date();
    // warm up the model
    model.predict(tf.ones([1, INPUT_NODE])).dispose();
    end = new Date();
    if (backend == "gpu")
        warmupTime = end - start;
}

async function infer(data){
    await triggerStart();
    statusLog("Inferring");

    let batch = data.nextTestBatch(inferSize);
    console.log("infer" + inferSize);
    console.log(batch.xs.length);
    
    for (let i = 0; i < batch.xs.length/INPUT_NODE; i++){
        let input = batch.xs.slice(i * INPUT_NODE, (i + 1) * INPUT_NODE);
        let inputTensor = tf.tensor2d(input, [1, INPUT_NODE]);
        
        if (verbose)
            console.log("Case " + i);

        let begin = new Date();

        model.predict(inputTensor);

        let end = new Date();
        
        inferTime += end - begin;
    }

    triggerEnd(task + loadTime + "\t" + warmupTime + "\t" + inferTime);
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
    await infer(data);
    statusLog("Finished");
}
main();

