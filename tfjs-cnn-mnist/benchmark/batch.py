import os

# trainsize = 6400
# infersize = 10000
units = [1, 2, 4, 8]
filters = [8, 16, 32]
batchsize = [8, 16, 32]

for i in units:
    for j in filters:
        for k in batchsize:
            os.system("python3 benchmark.py -u %d -f %d -b %d -t 6400 -i 10000" % (i, j, k))
            # python# benchmark.py -u 1 -f 64 -b 64 -t 6400 -i 10000
