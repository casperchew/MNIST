import NN

model = NN.NN([
    NN.LinearLayer(784, 300),
    NN.SigmoidLayer(),
    NN.LinearLayer(300, 10),
    NN.SigmoidLayer(),
    NN.SoftmaxLayer()
])

model.save('784-300-10')