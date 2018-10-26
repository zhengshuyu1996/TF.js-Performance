'use strict'
/*
author: David Xiang
email: xdw@pku.edu.cn
 */

let model, trainer;
let loadTime = 0;
let inferTime = 0;
let warmupTime = "cpu";

async function initModel(){
    if (verbose){
        console.log("init model");
    }

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
    loadTime = end - start;
    
    // warm up the model
    let testInput = new Float32Array(INPUT_NODE);
    start = new Date();
    model.predict({input: testInput});
    end = new Date();
    if (backend == "gpu")
        warmupTime = end - start;

}

async function infer(data){
    await triggerStart();
    statusLog("Inferring");

    let size = 100;
    let batch = data.nextTestBatch(size);
    let round = 0;

    while(inferTime < totTime){
        let index = round % size;
        let input = {
            input: batch.xs.slice(index * INPUT_NODE, (index + 1) * INPUT_NODE)
        }

        if (verbose)
            console.log("Case " + round);

        let begin = new Date();

        model.predict(input);
        
        let end = new Date();
        inferTime += end - begin;
        round++;
    }
    triggerEnd(task + loadTime + "\t" + warmupTime + "\t" + inferTime/round);
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

