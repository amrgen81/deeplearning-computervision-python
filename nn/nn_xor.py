# import the necessary packages
from neuralnetwork import NeuralNetwork
import numpy as np


# construct the XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# define our 2-2-1 neural network an train it
# input layer with two nodes (x has 2 features, e.g: [0, 0])
# single hidden layer with two nodes
# output layer with one node (i.e, 0 or 1)
nn = NeuralNetwork([2, 2, 2, 1], alpha=0.5)
nn.fit(X, y, epochs=20000)

# now that our network is trained, loop over the XOR data points
for (x, target) in zip(X, y):
    # make a prediction on the data point and display the result
    # to our console
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print("[INFO] data={}, groud-truth={}, pred={:.4f}, step={}".format(
        x, target[0], pred, step))

