'use strict'
/*
author: David Xiang
email: xdw@pku.edu.cn
 */

// constant args
const LOCALHOST = "http://localhost:8000";
const IMAGE_LENGTH = 28;
const INPUT_NODE = 784;
const OUTPUT_NODE = 10;
const NUM_CHANNELS = 1;
const LEARNING_RATE = 0.15;
const BATCH_SIZE = 64;

// args to be extracted from url
let libName;
let task;
let backend;
let trainSize;
let hiddenLayerSize;
let hiddenLayerNum;

let trainBatch;
let verbose = true;
let dotest = false;

let libList = ["tensorflowjs", "brainjs", "synaptic", "convnetjs"];
let backendList = ["cpu", "gpu"];

function getParam(query, key){
    let regex = new RegExp(key+"=([^&]*)","i");
    return query.match(regex)[1];
}


function parseArgs(){
    let address = document.location.href;
    let query = address.split("?")[1];

    libName = getParam(query, "libname");
    backend = getParam(query, "backend");
    trainSize = parseInt(getParam(query, "trainsize"));
    hiddenLayerSize = parseInt(getParam(query, "hiddenlayersize"));
    hiddenLayerNum = parseInt(getParam(query, "hiddenlayernum"));
    trainBatch = trainSize / BATCH_SIZE;


    // check whether these params are valid
    if (backendList.indexOf(backend) === -1){
        triggerStart();
        triggerEnd("Invalid URI:" + address);
        console.error("Invalid URI:" + address);
        return false;
    }

    if (libList.indexOf(libName) === -1){
        triggerStart();
        triggerEnd("Invalid URI:" + address);
        console.error("Invalid URI:" + address);
        return false;
    }

    if (trainSize <= 0 || hiddenLayerSize <= 0 || hiddenLayerNum <= 0){
        triggerStart();
        triggerEnd("Invalid URI:" + address);
        console.error("Invalid URI:" + address);
        return false;
    }

    

    // get right task name
    task = "jslib\ttraining\tmnist\t" + libName + "\t" + backend + "\t" 
    + trainSize + "\t" + hiddenLayerNum + "\t" + hiddenLayerSize + "\t";
    document.getElementById("task").innerText = task;
    return true;
}