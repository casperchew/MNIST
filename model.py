import ANN

model = ANN.ANN([
    ANN.LinearLayer(784, 300),
    ANN.SigmoidLayer(),
    ANN.LinearLayer(300, 10),
    ANN.SigmoidLayer(),
    ANN.SoftmaxLayer()
])

model.save('784-300-10')