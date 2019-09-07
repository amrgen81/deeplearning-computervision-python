# import the necessary packages
import numpy as np
import cv2

# initialize the class labels and set the seed of the pseudorandom
# number generator so we can reproduce our results
labels = ["dog", "cat", "panda"]
np.random.seed(1)

# randomly initialize our weight matrix and bias vector -- in a
# *real* training and classification task, these parameters would
# be *learned* by our model, but for the sake of this example,
# let's use random values

# the weight matrix has 3 rows (one for each of the class labels)
# and 3072 columns (one for each of the image height-width-channel 32x32x3 pixels)
W = np.random.randn(3, 3072)
# the bias matrix has 3 rows (corresponding to the number of class labels)
# along with one column
b = np.random.randn(3)

# load our example image, resize it and then flatten it into
# our "feature vector" representation
orig = cv2.imread("cats_00229.jpg")
image = cv2.resize(orig, (32,32)).flatten()

# compute the output scores by taking the dot product between the
# weight matrix and image pixels (X matrix), followed by adding the bias
scores = W.dot(image) + b  # scoring function

# loop over the scores + labels and display them
for (label, score) in zip(labels, scores):
    print("[INFO] {}: {:.2f}".format(label, score))

# draw the label with the highest score on the image as our prediction
cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# display our input image
# print(orig[0])
cv2.imshow("Image", orig)
cv2.waitKey(0)
