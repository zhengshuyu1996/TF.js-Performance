'use strict'
/*
author: David Xiang
email: xdw@pku.edu.cn
 */

// constant args
const LOCALHOST = "http://localhost:8000";
// const LOCALPATH = "/Users/xiangdongwei/TF.js-Performance";
const IMAGE_LENGTH = 28;
const INPUT_NODE = 784;
const OUTPUT_NODE = 10;
const NUM_CHANNELS = 1;

// args to be extracted from url
let libName;
let task;
let backend;
let inferSize;
let hiddenLayerSize;
let hiddenLayerNum;

let verbose = false;

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
    inferSize = parseInt(getParam(query, "infersize"));
    hiddenLayerSize = parseInt(getParam(query, "hiddenlayersize"));
    hiddenLayerNum = parseInt(getParam(query, "hiddenlayernum"));

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

    if (inferSize <= 0 || hiddenLayerSize <= 0 || hiddenLayerNum <= 0){
        triggerStart();
        triggerEnd("Invalid URI:" + address);
        console.error("Invalid URI:" + address);
        return false;
    }

    

    // get right task name
    task = "jslib\tinference\tmnist\t" + libName + "\t" + backend + "\t" 
    + inferSize + "\t" + hiddenLayerNum + "\t" + hiddenLayerSize + "\t";
    document.getElementById("task").innerText = task;
    return true;
}
