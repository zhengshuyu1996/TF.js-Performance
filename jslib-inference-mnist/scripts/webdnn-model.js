'use strict'
/*
author: David Xiang
email: xdw@pku.edu.cn
 */
let model;
let inferTime = 0, loadTime = 0, warmupTime = "cpu";

async function initModel(){
    if (verbose){
        console.log("init model");
    }

    let path = LOCALHOST+"/model/webdnn/mnist-" + hiddenLayerNum + "-" + hiddenLayerSize;
    let bk;
    if (backend == "cpu"){
        bk = 'webassembly';
    }else{
        bk = 'webgl';
    }

    let start = new Date();
    
    // load models using webdnn's runner api
    // https://mil-tokyo.github.io/webdnn/docs/tutorial/keras.html
    model = await WebDNN.load(path, {backendOrder:[bk]});
    let end = new Date();
    loadTime = end - start;
    
    if (verbose){
        console.log(model.backendName);
    }
    
    // get input variable reference
    let x = model.inputs[0];

    // warm up the model
    let testInput = new Float32Array(INPUT_NODE);

    start = new Date();

    // warm up the model
    x.set(testInput);
    await model.run();

    end = new Date();
    if (backend == "gpu")
        warmupTime = end - start;
}

async function infer(data){
    await triggerStart();
    statusLog("Inferring");

    // get input variable reference
    let x = model.inputs[0];

    let size = 100;
    let batch = data.nextTestBatch(size);
    
    let round = 0;
    while(inferTime < totTime){
        let index = round % size;
        let input = batch.xs.slice(index * INPUT_NODE, (index + 1) * INPUT_NODE);
        
        if (verbose)
            console.log("Case " + round);
        let begin = new Date();

        x.set(input);
        await model.run();

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