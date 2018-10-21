#!/bin/sh
for i in $(find *.h5)
do
    filename=$(echo $i | cut -d . -f1)
    echo $filename
    tensorflowjs_converter --input_format=keras $i $filename 
done
