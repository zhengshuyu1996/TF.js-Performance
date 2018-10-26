import os

trainsize = 640
infersize = 100000
hiddenlayernum = [1, 2, 4, 8, 16]
hiddenlayersize = [64, 128, 256, 512]
batchsize = [32, 64, 128]

for i in hiddenlayernum:
    for j in hiddenlayersize:
        for k in batchsize:
            os.system("python3 benchmark.py -n %d -s %d -b %d -t 640 -i 100000" % (i, j, k))
            # python# benchmark.py -n 1 -s 64 -b 64 -t 640 -i 100000