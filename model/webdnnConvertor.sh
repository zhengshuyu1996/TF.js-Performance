#!/bin/sh
pip3 install tensorflowjs
for i in $(find *.h5)
do
    filename=$(echo $i | cut -d . -f1)
    echo $filename
    python3 webdnn/webdnn/bin/convert_keras.py $i --input_shape '(1, 784, 1)' --out $filename
done
