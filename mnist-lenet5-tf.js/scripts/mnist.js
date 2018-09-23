/*
author: David Xiang
email: xdw@pku.edu.cn
 */
let btn = document.getElementById("train");
let equation = document.getElementById("equation");
equation.innerText = "total num of layers: (1 + 1) * 2 + 3 = 7";


let LEARNING_RATE = 0.1;
let optimizer = tf.train.sgd(LEARNING_RATE);
let BATCH_SIZE = 64;
let TRAIN_BATCHES = 1500;
let TEST_BATCH_SIZE = 1000;
let TEST_ITERATION_FREQUENCY = 5;
let IMAGE_LENGTH = 28;
let INPUT_NODE = 784;
let OUTPUT_NODE = 10;
let NUM_CHANNELS = 1;

let CONV1_SIZE = 5;
let CONV1_DEEP = 6;
let CONV2_SIZE = 5;
let CONV2_DEEP = 16;
let FLATTEN = 784;
let DENSE1_SIZE = 120;
let DENSE2_SIZE = 84;

function truncatedNormalTensor(shape){
    return tf.variable(tf.truncatedNormal(shape, mean=0, stddev=0.1)); // mean & stddev的位置不能调换？！
}
function zeroTensor(shape){
    return tf.variable(tf.zeros(shape));
}
let conv1Weights = truncatedNormalTensor(
        [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP]);
//let conv1Biases = zeroTensor([CONV1_DEEP]);

let conv2Weights = truncatedNormalTensor(
        [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP]);
//let conv1Biases = zeroTensor([CONV2_DEEP]);

let dense1Weights = truncatedNormalTensor([FLATTEN, DENSE1_SIZE]);
let dense1Biases = zeroTensor([DENSE1_SIZE]);
let dense2Weights = truncatedNormalTensor([DENSE1_SIZE, DENSE2_SIZE]);
let dense2Biases = zeroTensor([DENSE2_SIZE]);
let dense3Weights = truncatedNormalTensor([DENSE2_SIZE, OUTPUT_NODE]);
let dense3Biases = zeroTensor([OUTPUT_NODE]);


function inference(input_tensor){
    const xs = input_tensor.as4D(-1, IMAGE_LENGTH, IMAGE_LENGTH, NUM_CHANNELS);
    // conv1-pool1
    let conv1 = tf.tidy(()=>{
        return xs.conv2d(
                conv1Weights, strides=[1, 1, 1, 1], pad="same")
                //.add(tf.conv1Biases)
                .relu()
                .maxPool(2, 2, pad="same");
    });

    // conv2-pool2
    let conv2 = tf.tidy(()=>{
        return conv1.conv2d(
                conv2Weights, strides=[1, 1, 1, 1], pad="same")
                //.add(tf.conv2Biases)
                .relu()
                .maxPool(2, 2, pad="same");
    });

    let flatten = tf.tidy(()=>{
        return conv2.as2D(-1, FLATTEN);
    });

    // dense
    let dense1 = tf.tidy(()=>{
        return flatten.matMul(dense1Weights)
                      .add(dense1Biases)
                      .relu();
    });    
    let dense2 = tf.tidy(()=>{
        return dense1.matMul(dense2Weights)
                     .add(dense2Biases)
                     .relu();
    });    
    let dense3 = tf.tidy(()=>{
        return dense2.matMul(dense3Weights)
                     .add(dense3Biases);
    });

    return dense3;
}

function loss(labels, ys){
    return tf.losses.softmaxCrossEntropy(labels, ys).mean();
}

async function train(){
    let returnCost = true;

    for (let i = 0; i < TRAIN_BATCHES; i++){
        let cost = optimizer.minimize(()=>{
            let batch = data.nextTrainBatch(BATCH_SIZE);
            return loss(batch.labels, inference(batch.xs));
        }, returnCost);

        if(i % TEST_ITERATION_FREQUENCY === 0){
            let testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
            testBatch.xs = testBatch.xs.reshape(
                [TEST_BATCH_SIZE, IMAGE_LENGTH, IMAGE_LENGTH, NUM_CHANNELS]);
            let ys = inference(testBatch.xs);
            let correct_prediction = tf.equal(tf.argMax(ys, 1), tf.argMax(testBatch.labels, 1));
            let accuracy = tf.mean(tf.cast(correct_prediction, "float32"));
            infoLog("Batch #" + i + "    Loss: " + cost.dataSync() + "    Accuracy: " + accuracy.dataSync());
            tf.dispose(testBatch);
        }else{
            infoLog("Batch #" + i + "    Loss: " + cost.dataSync());
        }

        await tf.nextFrame();
    }
} 


let data;
async function load(){
    data = new MnistData();
    await data.load();
    statusLog("Ready");
}

btn.onclick = async function(){
    console.log("start training");
    console.time("train");
    await train();
    console.timeEnd("train");

    statusLog("Finished");
    console.log(tf.getBackend());
};

load();
