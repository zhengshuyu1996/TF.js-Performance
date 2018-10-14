/*
author: David Xiang
email: xdw@pku.edu.cn
 */
'use strict'
let model;
const TASK = "Inference\tkerasjs\tgpu\t";

async function initModel(){
    // load models
    model = new KerasJS.Model({
        filepath: LOCAL_SERVER+"/model/kerasjs/model.bin",
        gpu: true
    });

    // wait until model is ready
    await model.ready();
    //console.log(model.modelConfig);
    
    // warm up the model
    let testInput = new Float32Array(INPUT_NODE);
    for (let i = 1; i < 10; i++)
        model.predict({input: testInput});
}

async function infer(data){
    await triggerStart();
    statusLog("Inferring");

    let totTime = 0;
    let batch = data.nextTestBatch(TEST_SIZE);
    for (let i = 0; i < batch.labels.length/OUTPUT_NODE; i++){
        let input = {
            input: batch.xs.slice(i * INPUT_NODE, (i + 1) * INPUT_NODE)
        }

        if (VERBOSE)
            console.log("Case " + i);

        let begin = new Date();

        model.predict(input);
        
        let end = new Date();
        totTime += end - begin;
    }
    triggerEnd(TASK + totTime + "ms\t");
}

async function init(){
    await initModel();

    let data = new MnistData();
    await data.load();
    
    statusLog("Ready");
    return data;
}

async function main(){
    let data = await init();
    await infer(data);
    statusLog("Finished");
}
main();

