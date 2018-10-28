path = "http://localhost:8000/tfjs-lstm-inference/index.html"

backend_list = ["cpu", "gpu"]
lstmLayerSizes = [1, 2, 3, 4]

print("[")

r1 = len(backend_list)
r2 = len(lstmLayerSizes)
count = 0

for i in range(r1):
    for j in range(r2):
            print('    "%s?&backend=%s&layersizes=%d"' % 
                    (path, backend_list[i], lstmLayerSizes[j]), end="")
            if (count != r1 * r2 - 1):
                print(",")
            else:
                print("")
            count += 1

print("]")
