/*
author: David Xiang
email: xdw@pku.edu.cn
 */
'use strict'
let model;
const TASK = "Inferrence\ttfjs\tgpu\t";
async function initModel(){
    tf.setBackend("webgl");
    console.log(tf.getBackend());
    // load models
    model = await tf.loadModel(LOCAL_SERVER+"/model/tfjs/model.json");

    // warm up the model
    for (let i = 1; i < 10; i++)
        model.predict(tf.ones([1, INPUT_NODE])).dispose();
}

async function infer(data){
    await triggerStart();
    statusLog("Inferring");

    let totTime = 0;
    let batch = data.nextTestBatch(TEST_SIZE);
    
    for (let i = 0; i < batch.labels.length/OUTPUT_NODE; i++){
        let input = tf.tensor2d(batch.xs.slice(i * INPUT_NODE, (i + 1) * INPUT_NODE), [1, INPUT_NODE]);
        
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


