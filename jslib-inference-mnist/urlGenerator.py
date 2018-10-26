path = "http://localhost:8000/jslib-inference-mnist/"

cpu_list = [
    "convnetjs", "synaptic", "brainjs"
]

gpu_list = [
    "webdnn", "kerasjs", "tensorflowjs"
]

backend_list = ["cpu", "gpu"]
processtime = 10000
hiddenlayernum = [1, 2, 4]
hiddenlayersize = [64, 128, 256]


print("[")

r1 = len(gpu_list)
r2 = len(backend_list)
r3 = len(hiddenlayernum)
r4 = len(hiddenlayersize)
count = 0

for i in range(r1):
    for j in range(r2):
        for k in range(r3):
            for l in range(r4):
                print('    "%s%s.html?libname=%s&backend=%s&processtime=%d&hiddenlayernum=%d&hiddenlayersize=%d",' % 
                    (path, gpu_list[i], gpu_list[i], backend_list[j], processtime, hiddenlayernum[k], hiddenlayersize[l]))
                count += 1

r1 = len(cpu_list)
r2 = len(hiddenlayernum)
r3 = len(hiddenlayersize)

for i in range(r1):
    for j in range(r2):
        for k in range(r3):
            print('    "%s%s.html?libname=%s&backend=cpu&processtime=%d&hiddenlayernum=%d&hiddenlayersize=%d"' % 
                    (path, cpu_list[i], cpu_list[i], processtime, hiddenlayernum[j], hiddenlayersize[k]), end="")
            if (count !=  9 * r2 * r3 - 1):
                print(",")
            else:
                print("")
            count += 1
print("]")
