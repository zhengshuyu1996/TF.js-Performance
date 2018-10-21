# Model

All the pretrained model goes here. `TensorFlow.js`, `Kerasjs`, `WebDNN` all require converting the original `.h5` keras model. Special dependency and shell command are as follows.

### Tensorflow.js

You need to install `tensorflowjs` through `pip` and then type in:

```shell
tensorflowjs_converter --input_format=keras modelname.h5 dir
```

### Kerasjs

You need to clone Kerasjs’s GitHub repo to this dir and then type in:

```shell
python3 keras-js/python/encoder.py modelname.h5 -n NAME
```

The arg `NAME` is the name converted model file.

### WebDNN

You need to clone WebDNN’s Github repo to this dir. And then follow its tutorial [Setup guide (for Mac / Linux)](https://mil-tokyo.github.io/webdnn/docs/tutorial/setup.html#) CAREFULLY to install some dependent tools and add path to shell.

Then you might try entering this to generate your model:

```shell 
python3 webdnn/bin/convert_keras.py modelname.h5 --input_shape INPUTSHAPE --out NAME
```

The arg `INPUTSHAPE` are the input shape of original keras model which is like `'(1,224,224,3)'`.

 