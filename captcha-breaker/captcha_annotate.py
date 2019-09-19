import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import zipfile
import argparse
import os
from imutils import paths
from captcha_helper import CaptchaHelper
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input directory of captcha images")
ap.add_argument("-a", "--annot", required=True,
                help="path to output directory of annotations, "
                     "note: the annotated file list'll be saved in annotated.list")
args = vars(ap.parse_args())

# grab the image paths then initialize the dictionary of character counts
imagePaths = list(paths.list_images(args["input"]))
counts = {}

# load the last-annotated images so we will not annotate the image twice
# the index file path is {annotation_folder}/annotated.list
annotatedList = []
annotatedFile = "{}/{}".format(args["annot"], "annotated.list")

# if the output directory does not exist, create it
if os.path.isfile(annotatedFile):
    with open(annotatedFile, 'rb') as fp:
        annotatedList = pickle.load(fp)

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # check if image has been annotated
    imageName = imagePath.split(os.path.sep)[-1]
    if imageName in annotatedList:
        print("[INFO] image already annotated {}".format(imageName))
        continue

    # display an update to the user
    print("[INFO] processing image {}/{}".format(i + 1,
                                                 len(imagePaths)))
    try:
        # load image and convert to grey
        img = cv2.imread(imagePath, 0)
        contours = CaptchaHelper.get_words(img)
        # loop over the contours
        for c in contours:
            # display the character, making it larger enough for us to see
            # then wait for a key press
            cv2.imshow("ROI", imutils.resize(c, width=28))
            key = cv2.waitKey(0)

            # if the tilde key ("'") is pressed, the ignore the character
            if key == ord("'"):
                print("[INFO] ignoring character")
                continue
            # grab the key that was pressed and construct the path
            # to the output directory
            key = chr(key).upper()
            dirPath = os.path.sep.join([args["annot"], key])
            # if the output directory does not exist, create it
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
            # write the labeled character to file
            count = counts.get(key, 1)
            p = os.path.sep.join([dirPath, "{}.png".format(str(count).zfill(6))])
            cv2.imwrite(p, c)

            # increase the count for the current key
            counts[key] = count + 1

        # save annotated image
        annotatedList.append(imageName)
    # we are trying to ctrl-c out of the script
    except KeyboardInterrupt:
        print("[INFO] mannually leaving script")
        break
    # an unknown error has occurred for this particular image
    except:
        print("[INFO] skipping image...")
        # save annotated image
        annotatedList.append(imageName)

# write the annotated file list
with open(annotatedFile, 'wb') as fp:
    pickle.dump(annotatedList, fp)
