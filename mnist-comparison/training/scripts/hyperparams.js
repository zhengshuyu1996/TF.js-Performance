const BATCH_SIZE = 64;
const TRAIN_SIZE = 200; // 60000
const TRAIN_BATCHES = TRAIN_SIZE / BATCH_SIZE;
const TEST_SIZE = 200; // 5000

const IMAGE_LENGTH = 28;
const INPUT_NODE = 784;
const HIDDEN_SIZE = 128;
const OUTPUT_NODE = 10;
const NUM_CHANNELS = 1;

const LEARNING_RATE = 0.15;

const LOCAL_SERVER = "http://localhost:8000";

const VERBOSE = true;
const DO_TEST = false;