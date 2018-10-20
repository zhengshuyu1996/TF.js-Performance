html_list = [
    "http://localhost:8000/tfjs-inference-keras/template.html"
]

model_list = [
    "densenet121", "mobilenet", "nasnetlarge", "vgg16", "densenet169",
    "mobilenetv2", "resnet50", "vgg19", "xception", "inceptionv3", "inceptionresnetv2"
];

backend_list = ["cpu", "gpu"]

testsize = 2000

print("[")

r1 = len(html_list)
r2 = len(model_list)
r3 = len(backend_list)
count = 0

for i in range(r1):
    for j in range(r2):
        for k in range(r3):
            print('    "%s?model=%s&backend=%s&testsize=%d"' % 
                    (html_list[i], model_list[j], backend_list[k], testsize), end="")
            if (count != r1 * r2 * r3 - 1):
                print(",")
            else:
                print("")
            count += 1

print("]")
