import yaml

import NN

config = yaml.safe_load(open('config.yaml'))

model = NN.NN([
    NN.LinearLayer(784, 300),
    NN.SigmoidLayer(),
    NN.LinearLayer(300, 10),
    NN.SigmoidLayer(),
    NN.SoftmaxLayer()
])

model.save(config['model'])