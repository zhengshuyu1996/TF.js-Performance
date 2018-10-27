import os

trainsize = 6400
hiddenlayernum = [1, 2, 4, 8]
hiddenlayersize = [64, 128, 256]

for i in hiddenlayernum:
    for j in hiddenlayersize:
        os.system("python3 benchmark.py -t 6400 -n %d -s %d" % (i, j))