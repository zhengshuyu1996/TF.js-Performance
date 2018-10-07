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

    // get input variable reference
    let x = model.inputs[0];

    for (let i = 0; i < batch.labels.length/OUTPUT_NODE; i++){;
        let input = batch.xs.slice(i * INPUT_NODE, (i + 1) * INPUT_NODE);
        x.set(input);
        await model.run();

        /*let predictlabel = WebDNN.Math.argmax(model.outputs[0])[0];
        let truth = WebDNN.Math.argmax(
            batch.labels.slice(i * OUTPUT_NODE, (i + 1) * OUTPUT_NODE)
            )[0];
        if (truth === predictlabel)
            count+=1;*/
    }
    console.timeEnd("inference");
}

async function load(){
    let data = new MnistData();
    await data.load();

    // load models using webdnn's runner api
    // https://mil-tokyo.github.io/webdnn/docs/tutorial/keras.html
    model = await WebDNN.load(LOCAL_SERVER + "/model/webdnn",
        {backendOrder:['webassembly']});
    console.log(model.backendName);
    
    // get input variable reference
    let x = model.inputs[0];

    // warm up the model
    let testInput = new Float32Array(INPUT_NODE);
    for (let i = 1; i < 10; i++){
        x.set(testInput);
        await model.run();
    }
    
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

