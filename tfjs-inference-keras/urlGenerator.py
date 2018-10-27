html_list = [
    "http://localhost:8000/tfjs-inference-keras/template.html"
]

model_list = [
    "densenet121", "mobilenetv2", "resnet50", "xception", "inceptionv3"
]; # others nasnetlarge mobilenet desenet169 vgg16/19 desnet169 bug?

backend_list = ["cpu", "gpu"]

infer_time = 60000

print("[")

r1 = len(html_list)
r2 = len(model_list)
r3 = len(backend_list)
count = 0

for i in range(r1):
    for j in range(r2):
        for k in range(r3):
            if backend_list[k] == "gpu":
                testsize = 1000
            else:
                testsize = 15
            print('    "%s?model=%s&backend=%s&processtime=%d"' % 
                    (html_list[i], model_list[j], backend_list[k], infer_time), end="")
            if (count != r1 * r2 * r3 - 1):
                print(",")
            else:
                print("")
            count += 1

print("]")
