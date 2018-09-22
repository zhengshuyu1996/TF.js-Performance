/*
author: David Xiang
email: xdw@pku.edu.cn
 */
console.log(tf.getBackend());
//tf.setBackend("cpu");
var btn = document.getElementById("train");
var nConv = document.getElementById("nConv");
var nPool = document.getElementById("nPool");
var equation = document.getElementById("equation");
equation.innerText = "total num of layers: (1 + 1) * 2 + 3 = 7";
var mPerform = document.getElementById("mPerform");
var mInfo;

const LEARNING_RATE = 0.15;
const optimizer = tf.train.sgd(LEARNING_RATE);
const BATCH_SIZE = 64;
const TRAIN_BATCHES = 150;
const TEST_BATCH_SIZE = 1000;
const TEST_ITERATION_FREQUENCY = 5;
const IMAGE_LENGTH = 28;
const INPUT_NODE = 784;
const OUTPUT_NODE = 10;
const NUM_CHANNELS = 1;

const CONV1_SIZE = 5;
const CONV1_DEEP = 6;
const CONV2_SIZE = 5;
const CONV2_DEEP = 16;
const FLATTEN = 784;
const DENSE1_SIZE = 120;
const DENSE2_SIZE = 84;

function truncatedNormalTensor(shape){
    return tf.variable(tf.truncatedNormal(shape, stddev=0.1));
}
function zeroTensor(shape){
    return tf.variable(tf.zeros(shape));
}
const conv1Weights = truncatedNormalTensor(
        [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP]);
//const conv1Biases = zeroTensor([CONV1_DEEP]);

const conv2Weights = truncatedNormalTensor(
                [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP]);
//const conv1Biases = zeroTensor([CONV2_DEEP]);

const dense1Weights = truncatedNormalTensor([FLATTEN, DENSE1_SIZE]);
const dense1Biases = zeroTensor([DENSE1_SIZE]);
const dense2Weights = truncatedNormalTensor([DENSE1_SIZE, DENSE2_SIZE]);
const dense2Biases = zeroTensor([DENSE2_SIZE]);
const dense3Weights = truncatedNormalTensor([DENSE2_SIZE, OUTPUT_NODE]);
const dense3Biases = zeroTensor([OUTPUT_NODE]);


function inference(input_tensor, regularizer){
    // conv1-pool1
    const conv1 = tf.tidy(()=>{
        return input_tensor.conv2d(
                conv1Weights, strides=[1, 1, 1, 1], pad="same")
                //.add(tf.conv1Biases)
                .relu()
                .maxPool(2, 2, pad="same");
    });

    // conv2
    const conv2 = tf.tidy(()=>{
        return conv1.conv2d(
                conv2Weights, strides=[1, 1, 1, 1], pad="same")
                //.add(tf.conv2Biases)
                .relu()
                .maxPool(2, 2, pad="same");
    });

    const flatten = conv2.as2D(-1, FLATTEN);    

    // dense
    const dense1 = flatten.matMul(dense1Weights)
                          .add(dense1Biases)
                          .relu();
    const dense2 = dense1.matMul(dense2Weights)
                         .add(dense2Biases)
                         .relu();
    const dense3 = dense2.matMul(dense3Weights)
                        .add(dense3Biases);

    return dense3;
}
async function train(){
    const returnCost = true;

    for (let i = 0;i < TRAIN_BATCHES; i++){
        let cost = optimizer.minimize(()=>{
            const batch = data.nextTrainBatch(BATCH_SIZE);
            batch.xs = batch.xs.reshape(
                [BATCH_SIZE, IMAGE_LENGTH, IMAGE_LENGTH, NUM_CHANNELS]);
            let ys = inference(batch.xs);
            return tf.losses.softmaxCrossEntropy(batch.labels, ys).mean() 
        }, returnCost);


        if(i % TEST_ITERATION_FREQUENCY == 0){
            let testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
            testBatch.xs = testBatch.xs.reshape(
                [TEST_BATCH_SIZE, IMAGE_LENGTH, IMAGE_LENGTH, NUM_CHANNELS]);
            let ys = inference(testBatch.xs);
            correct_prediction = tf.equal(tf.argMax(ys, 1), tf.argMax(testBatch.labels, 1));
            accuracy = tf.mean(tf.cast(correct_prediction, "float32"));
            console.log(typeof(accuracy))
            infoLog("Batch #" + i + "    Loss: " + cost.dataSync() + "    Accuracy: " + accuracy.dataSync());

        }else{
            infoLog("Batch #" + i + "    Loss: " + cost.dataSync());
        }

        if (i == parseInt(TRAIN_BATCHES / 2)){
            mInfo = tf.memory();
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

    mPerform.innerText = "memory performance: " + JSON.stringify(mInfo);
    statusLog("Finished");
    console.log(tf.getBackend());
};

load();
