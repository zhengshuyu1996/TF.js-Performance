# TF.js-Performance
Measuring performance of TensorFlow.js on various dimensions

### Lenet-5 on MNIST

Batch size: 64

Training data set: 60000 pictures with 28*28 pixels

Epochs: 8

Evaluation size: 1000 (evaluation is done after each epoch)

| Platform            | API   | Round 1(s) | Round 2(s) | Round 3(s) | Round 4(s) |
| ------------------- | ----- | ---------- | ---------- | ---------- | ---------- |
| TensorFlow (Python) | Core  |            |            |            |            |
| TensorFlow (Python) | Keras |            |            |            |            |
| TensorFlow.js       | Core  |            |            |            |            |
| TensorFlow.js       | Keras |            |            |            |            |

