/*
author: David Xiang
email: xdw@pku.edu.cn
 */
'use strict'
let model;
let task;// = "Tfjs\tres50\tcpu\t";
let modelName;
let backend;
let testsize;

let backendList = ["cpu", "gpu"];
let model224List = [
    "densenet121", "mobilenet", "nasnetlarge", "vgg16", "densenet169",
    "mobilenetv2", "resnet50", "vgg19"
];
let model299List = ["xception", "inceptionv3", "inceptionresnetv2"];


function parseArgs(){
    let address = document.location.href;
    let query = address.split("?")[1];
    let args = query.split("&");

    return true;
}
async function initData(){
    let offset = tf.scalar(127.5);
    let data = [];
    for (let i = 1; i <= DATA_SIZE; i++){
        // load img from a canvas
        let img = document.getElementById("pic"+i);
        let dataItem = tf.fromPixels(img).
                        toFloat().
                        sub(offset).
                        div(offset).
                        reshape([1, 224, 224, 3]);
        data.push(dataItem);
    }
    return data;
}
async function initModel(){
    //set backend
    tf.setBackend("cpu");
    console.log(tf.getBackend());

    // load model
    console.log("loading model");
    statusLog("Loading Model");
    model = await tf.loadModel(LOCAL_SERVER+"/model/tfjs/resnet50/model.json");

    // warm up the model
    console.log("warmup");
    statusLog("Warming up");
    for (let i = 0; i < 3; i++){
        console.time("warmup");
        model.predict(tf.zeros([1, 224, 224, 3])).dispose();
        console.timeEnd("warmup");
    }
}

async function infer(data){
    triggerStart();
    statusLog("Inferring");

    let totTime = 0;
    for (let i = 0; i < TEST_SIZE; i++){
        if (VERBOSE)
            console.log("Case" + i);

        let begin = new Date();
        model.predict(data[i%DATA_SIZE]);
        let end = new Date();
        totTime += end - begin;
    }
    triggerEnd(TASK + "time:\t" + totTime + "ms\t");
}

async function init(){
    await initModel();

    let data = initData();

    statusLog("Ready");
    return data;
}


async function main(){
    let argsStatus = parseArgs();
    if argsStatus
    // TODO if parseing failed
    let data = await init();
    await infer(data);
    statusLog("Finished");
}
main();

