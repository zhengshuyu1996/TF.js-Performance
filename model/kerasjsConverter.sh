#!/bin/sh
for i in $(find *.h5)
do
    filename=$(echo $i | cut -d . -f1)
    echo $filename
    python3 keras-js/python/encoder.py $i
done
