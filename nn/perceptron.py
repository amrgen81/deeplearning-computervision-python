# import the necessary packages
import numpy as np


class Perceptron:
    def __init__(self, N, alpha=0.1):
        # N: the number of columns in our input feature vectors. In the context of our bitwise
        # dataset we'll set N equal to 2 since there are two inputs
        # alpha: our learning rate for the Perceptron algorithm. Common choices of learning
        # rates are normally in the range alpha = 0.1, 0.01, 0.001
        # initialize the weight matrix and store the learning rate
        self.W = np.random.rand(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        # apply the step function
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        # insert a column of 1's as the last entry in the feature
        # matrix -- this little trick allows us to treat the bias
        # as a trainable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # training procedure
        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # loop over each individual data point
            for (x, target) in zip(X, y):
                # take the dot product between the input features
                # and the weight matrix, then pass this value
                # through the step function to obtain the prediction
                p = self.step(np.dot(x, self.W))

                # only perform a weight update if our prediction
                # does not match the target
                if p != target:
                    # determine the error
                    error = p - target
                    # update the weight matrix
                    self.W += -self.alpha * error * x

    def predict(self, X, addBias=True):
        # predict the class labels for a given set of input data
        # ensure our input is a matrix
        X = np.atleast_2d(X)

        # check to see if the bias column should be added
        if addBias:
            # insert a column of 1's as the last entry in the feature matrix (bias)
            X = np.c_[X, np.ones((X.shape[0]))]

        # take the dot product between the input features and the
        # weight matrix, then pass the value through the step function
        return self.step(np.dot(X, self.W))
