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
const TEST_SIZE = 1000;

// args to be extracted from url
let kernelSize = 5; // fixed yet
let poolSize = 2; // fixed yet
let task;
let backend;
let trainTime;
let inferTime;
let units;
let filters;
let batchSize;

let verbose = true;
let dotest = false;

let backendList = ["cpu", "gpu"];

function getParam(query, key){
    let regex = new RegExp(key+"=([^&]*)","i");
    return query.match(regex)[1];
}


function parseArgs(){
    let address = document.location.href;
    let query = address.split("?")[1];

    backend = getParam(query, "backend");
    trainTime = parseInt(getParam(query, "traintime"));
    inferTime = parseInt(getParam(query, "infertime"));
    units = parseInt(getParam(query, "units"));
    filters = parseInt(getParam(query, "filters"));
    batchSize = parseInt(getParam(query, "batchsize"));


    // check whether these params are valid
    if (backendList.indexOf(backend) === -1){
        triggerStart();
        triggerEnd("Invalid URI:" + address);
        console.error("Invalid URI:" + address);
        return false;
    }

    if (trainTime <= 0 || inferTime <= 0 || units <= 0 || filters <= 0 || batchSize <= 0){
        triggerStart();
        triggerEnd("Invalid URI:" + address);
        console.error("Invalid URI:" + address);
        return false;
    }

    

    // get right task name
    task = "tfjs\tcnn\t" + backend + "\t" 
     + units + "\t" + filters + "\t" + batchSize + "\t";
    document.getElementById("task").innerText = task;
    return true;
}
