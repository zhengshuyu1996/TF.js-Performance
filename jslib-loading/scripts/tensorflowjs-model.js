'use strict'
/*
author: David Xiang
email: xdw@pku.edu.cn
 */

let model;
let round = 10;
let inferTime = 0, loadTime = 0, warmupTime = 0;

async function initModel(){
    await triggerStart();
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

    for (let i = 0; i < round; i++){
        let start = new Date();

        // load models
        let path = LOCALHOST+"/model/tensorflowjs/mnist-" + 
            + hiddenLayerNum + "-" + hiddenLayerSize + "/model.json";

        model = await tf.loadModel(path);

        let end = new Date();
        loadTime += end - start;

        
        start = new Date();
        // warm up the model
        model.predict(tf.ones([1, INPUT_NODE])).dispose();
        end = new Date();
        if (backend == "gpu")
            warmupTime += end - start;
    }
    warmupTime = warmupTime/round;
    if (backend == "cpu")
        warmupTime = "cpu";
    triggerEnd(task + loadTime/round + "\t" + warmupTime + "\t" + inferTime/round);
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
    statusLog("Finished");
}
main();

