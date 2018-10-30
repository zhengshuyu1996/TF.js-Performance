'use strict'
/*
author: David Xiang
email: xdw@pku.edu.cn
 */

let model, trainer;
let loadTime = 0;
let inferTime = 0;
let warmupTime = 0;
let round = 10;

async function initModel(){
    await triggerStart();
    if (verbose){
        console.log("init model");
    }

    for (let i = 0; i < round; i++){
        // load models
        let path = LOCALHOST+"/model/kerasjs/mnist-" + hiddenLayerNum + "-" + hiddenLayerSize + ".bin";

        let start = new Date();
        model = new KerasJS.Model({
            filepath: path ,
            gpu: false
        });

        // wait until model is ready
        await model.ready();
        //console.log(model.modelConfig);

        let end = new Date();
        loadTime += end - start;
        
        // warm up the model
        let testInput = new Float32Array(INPUT_NODE);
        start = new Date();
        model.predict({input: testInput});
        end = new Date();
        //console.log(start);
        //console.log(end);
        //console.log(warmupTime);
        if (backend == "gpu")
            warmupTime += end - start;
    }
    warmupTime = warmupTime/round;
    if (backend == "cpu")
        warmupTime = "cpu"
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

