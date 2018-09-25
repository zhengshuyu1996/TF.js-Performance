/*
author: David Xiang
email: xdw@pku.edu.cn
 */
let btn = document.getElementById("train");
let ckbx = document.getElementById("use gpu");

const BATCH_SIZE = 64;
const TRAIN_BATCHES = 7500; // total epochs = 8
const TEST_BATCH_SIZE = 1000;
const TEST_ITERATION_FREQUENCY = 1000;
const IMAGE_LENGTH = 28;
const INPUT_NODE = 784;
const OUTPUT_NODE = 10;
const NUM_CHANNELS = 1;

const LEARNING_RATE = 0.15;
const optimizer = tf.train.sgd(LEARNING_RATE);
const CONV1_SIZE = 5;
const CONV1_DEEP = 6;
const CONV2_SIZE = 5;
const CONV2_DEEP = 16;
const FLATTEN = 784;
const DENSE1_SIZE = 120;
const DENSE2_SIZE = 84;

function truncatedNormalTensor(shape){
    return tf.variable(tf.truncatedNormal(shape, mean=0, stddev=0.1)); // mean & stddev的位置不能调换？！
}
function zeroTensor(shape){
    return tf.variable(tf.zeros(shape));
}
const conv1Weights = truncatedNormalTensor(
        [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP]);
//let conv1Biases = zeroTensor([CONV1_DEEP]);

const conv2Weights = truncatedNormalTensor(
        [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP]);
//let conv1Biases = zeroTensor([CONV2_DEEP]);

const dense1Weights = truncatedNormalTensor([FLATTEN, DENSE1_SIZE]);
const dense1Biases = zeroTensor([DENSE1_SIZE]);
const dense2Weights = truncatedNormalTensor([DENSE1_SIZE, DENSE2_SIZE]);
const dense2Biases = zeroTensor([DENSE2_SIZE]);
const dense3Weights = truncatedNormalTensor([DENSE2_SIZE, OUTPUT_NODE]);
const dense3Biases = zeroTensor([OUTPUT_NODE]);


function inference(input_tensor){
    const xs = input_tensor.as4D(-1, IMAGE_LENGTH, IMAGE_LENGTH, NUM_CHANNELS);
    // conv1-pool1
    const conv1 = tf.tidy(()=>{
        return xs.conv2d(
                conv1Weights, strides=[1, 1, 1, 1], pad="same")
                //.add(tf.conv1Biases)
                .relu()
                .maxPool(2, 2, pad="same");
    });

    // conv2-pool2
    const conv2 = tf.tidy(()=>{
        return conv1.conv2d(
                conv2Weights, strides=[1, 1, 1, 1], pad="same")
                //.add(tf.conv2Biases)
                .relu()
                .maxPool(2, 2, pad="same");
    });

    const flatten = tf.tidy(()=>{
        return conv2.as2D(-1, FLATTEN);
    });

    // dense
    const dense1 = tf.tidy(()=>{
        return flatten.matMul(dense1Weights)
                      .add(dense1Biases)
                      .relu();
    });    
    const dense2 = tf.tidy(()=>{
        return dense1.matMul(dense2Weights)
                     .add(dense2Biases)
                     .relu();
    });    
    const dense3 = tf.tidy(()=>{
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
            
            console.log("Batch #" + i + "    Loss: " + cost.dataSync() + "    Accuracy: " + accuracy.dataSync());
            tf.dispose(testBatch);
        }/*else{
            infoLog("Batch #" + i + "    Loss: " + cost.dataSync());
        }*/

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
    if (ckbx.checked == true){
        tf.setBackend("webgl");
    }else{
        tf.setBackend("cpu");
    }
    console.log(tf.getBackend());
    console.log("start training");

    console.time("train");
    await train();
    console.timeEnd("train");

    statusLog("Finished");
};

load();
