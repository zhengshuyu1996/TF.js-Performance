path = "http://localhost:8000/tfjs-dnn-mnist/tensorflowjs.html"

lib = "tensorflowjs";

backend_list = ["cpu", "gpu"]
train_time = 1000
infer_time = 1000
hiddenlayernum = [1, 2, 4, 8, 16]
hiddenlayersize = [64, 128, 256, 512]
batchsize = [32, 64, 128]


print("[")

r1 = len(backend_list)
r2 = len(hiddenlayernum)
r3 = len(hiddenlayersize)
r4 = len(batchsize)
count = 0

for i in range(r1):
    for j in range(r2):
        for k in range(r3):
            for l in range(r4):
                print('    "%s?&backend=%s&traintime=%d&infertime=%d&hiddenlayernum=%d&hiddenlayersize=%d&batchsize=%d"' % 
                    (path, backend_list[i], train_time, infer_time, hiddenlayernum[j], hiddenlayersize[k], batchsize[l]), end="")
                if (count != r1 * r2 * r3 * r4 - 1):
                    print(",")
                else:
                    print("")
                count += 1

print("]")