path = "http://localhost:8000/tfjs-lstm-training/index.html"

backend_list = ["cpu", "gpu"]
lstmLayerSizes = ["2", "4", "8", "16", "2,2", "4,4", "8,8", "16,16", "2,2,2", "4,4,4", "8,8,8", "16,16,16"]
process_time = 10000
examplesPerEpoch = 2048
batchSize = [32, 64, 128, 256]


print("[")

r1 = len(backend_list)
r2 = len(lstmLayerSizes)
r3 = len(batchSize)
count = 0

for i in range(r1):
    for j in range(r2):
        for k in range(r3):
            print('    "%s?&backend=%s&layersizes=%s&batchSize=%d&processtime=%d&numEpochs=%d"' % 
                    (path, backend_list[i], lstmLayerSizes[j], batchSize[k], process_time, examplesPerEpoch), end="")
            if (count != r1 * r2 * r3 - 1):
                print(",")
            else:
                print("")
            count += 1

print("]")
