path = "http://localhost:8000/jslib-training-mnist/"

lib_list = [
    "tensorflowjs", "convnetjs", "synaptic", "brainjs"
]

backend_list = ["cpu", "gpu"]
process_time = 10000
hiddenlayernum = [1, 2, 4, 8]
hiddenlayersize = [64, 128, 256]


print("[")

r1 = len(lib_list)
r2 = len(backend_list)
r3 = len(hiddenlayernum)
r4 = len(hiddenlayersize)
count = 0

for i in range(r1):
    for j in range(r2):
        for k in range(r3):
            for l in range(r4):
                if lib_list[i] != "tensorflowjs" and backend_list[j] == "gpu":
                    continue
                print('    "%s%s.html?libname=%s&backend=%s&processtime=%d&hiddenlayernum=%d&hiddenlayersize=%d"' % 
                    (path, lib_list[i], lib_list[i], backend_list[j], process_time, hiddenlayernum[k], hiddenlayersize[l]), end="")
                if (count != 5 * r3 * r4 - 1):
                    print(",")
                else:
                    print("")
                count += 1

print("]")
