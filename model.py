import ANN

model = ANN.ANN([
    ANN.LinearLayer(784, 10),
    ANN.SigmoidLayer(),
    ANN.SoftmaxLayer()
])

model.save('784-10_SGD')