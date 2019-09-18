from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
                help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained smile detector CNN")
ap.add_argument("-i", "--image",
                help="path to input image")
args = vars(ap.parse_args())

# load the face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])

# resize the input image, convert it to grayscale, and then clone the
# original frame so we can draw on it later in the program
image = cv2.imread(args["image"])
image = imutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the input frame, then clone the frame so that
# we can draw on it
rects = detector.detectMultiScale(gray, scaleFactor=1.1,
    minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

# loop over the faces (as rectangles)
for (fX, fY, fW, fH) in rects:
    # extract the ROI of the face from the grayscale image,
    # resize it to fixed 28x28 pixels, and then prepare the
    # ROI for classification via the CNN
    roi = gray[fY:fY+fH, fX:fX + fW]
    roi = cv2.resize(roi, (28, 28))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    # determine the probabilities of both "smiling" and "not smiling",
    # then set the label accordingly
    (notsmiling, smiling) = model.predict(roi)[0]
    isSmile = smiling > notsmiling
    label = "Smiling" if isSmile else "Not Smiling"

    # display the label and bounding box rectangle on the output frame
    cv2.putText(image, label, (fX, fY-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if isSmile else (0, 0, 255), 2)
    cv2.rectangle(image, (fX, fY), (fX + fW, fY + fH),
                  (0, 255, 0) if isSmile else (0, 0, 255), 2)

# show our detected faces along with smiling/not smiling labels
cv2.imshow("Face", image)
cv2.waitKey()
