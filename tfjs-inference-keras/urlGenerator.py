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
print('    "",')

for html in html_list:
    for model in model_list:
        for backend in backend_list:
            print('    "%s?model=%s&backend=%s&testsize=%d",' % (html, model, backend, testsize))

print("]")
