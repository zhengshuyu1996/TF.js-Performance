/*
author: David Xiang
email: xdw@pku.edu.cn
 */
'use strict'
let model;
const TASK = "Inference\twebdnn\tcpu\t";

async function initModel(){
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
}

async function infer(data){
    await triggerStart();
    statusLog("Inferring");

    let totTime = 0;
    let batch = data.nextTestBatch(TEST_SIZE);

    // get input variable reference
    let x = model.inputs[0];

    let count = 0;
    for (let i = 0; i < batch.labels.length/OUTPUT_NODE; i++){;
        let input = batch.xs.slice(i * INPUT_NODE, (i + 1) * INPUT_NODE);
        
        if (VERBOSE)
            console.log("Case " + i);
        let begin = new Date();

        x.set(input);
        await model.run();

        let end = new Date();
        totTime += end - begin;

        /*let predictlabel = WebDNN.Math.argmax(model.outputs[0])[0];
        let truth = WebDNN.Math.argmax(
            batch.labels.slice(i * OUTPUT_NODE, (i + 1) * OUTPUT_NODE)
            )[0];
        if (truth === predictlabel)
            count+=1;*/
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