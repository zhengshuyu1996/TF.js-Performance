import os
layernum = [1, 2, 4]
layersize = [64, 128, 256]

for i in layernum:
    for j in layersize:
        os.system("python3 modelGenerator.py -n %d -s %d" % (i, j))
