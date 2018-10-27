path = "http://localhost:8000/tfjs-cnn-mnist/tensorflowjs.html"

lib = "tensorflowjs";

backend_list = ["cpu", "gpu"]
train_time = 10000
infer_time = 1000
units = [1, 2, 4, 8]
filters = [8, 16, 32]
batchsize = [8, 16, 32]


print("[")

r1 = len(backend_list)
r2 = len(units)
r3 = len(filters)
r4 = len(batchsize)
count = 0

for i in range(r1):
    for j in range(r2):
        for k in range(r3):
            for l in range(r4):
                print('    "%s?&backend=%s&traintime=%d&infertime=%d&units=%d&filters=%d&batchsize=%d"' % 
                    (path, backend_list[i], train_time, infer_time, units[j], filters[k], batchsize[l]), end="")
                if (count != r1 * r2 * r3 * r4 - 1):
                    print(",")
                else:
                    print("")
                count += 1

print("]")
