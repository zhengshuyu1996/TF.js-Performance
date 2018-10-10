/*
author: David Xiang
email: xdw@pku.edu.cn
 */
'use strict'
let model;

async function infer(data){
    statusLog("Inferring");

    console.time("inference");
    let batch = data.nextTestBatch(TEST_SIZE);
    for (let i = 0; i < batch.labels.length/OUTPUT_NODE; i++){
        let input = {
            input: batch.xs.slice(i * INPUT_NODE, (i + 1) * INPUT_NODE)
        }
        model.predict(input);
        //console.log(i);
    }
    console.timeEnd("inference");
}

async function load(){
    let data = new MnistData();
    await data.load();

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

