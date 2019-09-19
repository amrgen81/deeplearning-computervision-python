from keras.preprocessing.image import img_to_array
from keras.models import load_model
from .captcha_helper import CaptchaHelper
from imutils import paths
import numpy as np
import argparse
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input directory of captcha images")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained model")
args = vars(ap.parse_args())

# load the pre-trained network
print("[INFO] loading pre-trained network..")
model = load_model(args["model"])

# grab the input images
imagePaths = list(paths.list_images(args["input"]))

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # display an update to the user
    print("[INFO] processing image {}/{}".format(i + 1,
                                                 len(imagePaths)))
    img = cv2.imread(imagePath, 0)

    # Process image to segment words in the image
    # Otsu thresholding with Gaussian Blur
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    dilation3 = cv2.dilate(th3, kernel, iterations=1)
    erosion3 = cv2.erode(dilation3, kernel, iterations=1)

    kernel = np.ones((3, 1), np.uint8)
    dilation3 = cv2.dilate(erosion3, kernel, iterations=1)
    # contouring
    # get the individual letters
    x, y, w, h = 30, 12, 20, 38

    predictions = []
    # extract words
    for idx in range(5):
        roi = dilation3[y:y + h, x: x + w]
        x += w
        roi = CaptchaHelper.preprocess(roi, 28, 28)
        roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
        pred = model.predict(roi).argmax(axis=1)[0] + 1
        predictions.append(str(pred))

        # draw the prediction on the output image
        cv2.rectangle(img, (x-2, y-2),
                      (x + w + 4, y + h + 4), (0, 255, 0), 1)
        cv2.putText(img, str(pred), (x-5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # show the output image
    print("[INFO] captcha: {}".format("".join(predictions)))
    cv2.imshow("output", img)
    cv2.waitKey()
