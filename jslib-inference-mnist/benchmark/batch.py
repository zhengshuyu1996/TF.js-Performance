import os

infer_size = 50000 
hiddenlayernum = [1, 2, 4, 8]
hiddenlayersize = [64, 128, 256]

print("ATTENTION: infer_size are set to be 50000 instead of 1000 so that the inference time can be measured precisely in ms.")
for i in hiddenlayernum:
    for j in hiddenlayersize:
        os.system("python3 benchmark.py -i %d -n %d -s %d" % (infer_size, i, j))
