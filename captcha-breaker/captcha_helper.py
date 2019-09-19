import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import zipfile
import argparse
import os


# CREDIT: https://www.kaggle.com/fournierp/opencv-word-segmenting-on-captcha-images
class CaptchaHelper:
    @staticmethod
    def get_words(img):
        # here we will tale the image: a 5 letter word with some noise and remove the noise
        # we will attempt different methods and see which ones perform the best on the data
        # firstly, we will convert the image to black and white. We will use Thresholding to do so.
        # Adaptive thresholding will dtermine when to set the image to black or white relative to the pixel's environment.
        # That is useful given the different shades of grey in the image.
        # Otsu Thresholding will calculate a threshold value from the image histogram.
        # We will olso try applying a Blur to remove the noise on the image (the fading on the 4th letter)

        # from RGB to BW
        # Adaptive thresholding
        th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 2)

        # Otsu thresholding
        ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Otsu thresholding with Gaussian Blur
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # plot
        '''
        titles = ['Original', 'Adaptive', "Otsu", 'Gaussian + Otsu']
        images = [img, th, th2, th3]
        for i in range(4):
            plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()
        '''

        #############

        # Then we will try to remove the noise (the line that traverses the image)
        # We will perform Erosions and Dilations (because it is black on white, erosion dilates
        # and dilation erodes). These operations are Morphological Transformations: mathematical
        # operations performed on the image's pixels. They will traverse the image with a matrix
        # of NxM (3 by 3 in our case) and multiply the image with it and save the result.
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(th, kernel, iterations=1)
        dilation2 = cv2.dilate(th2, kernel, iterations=1)
        dilation3 = cv2.dilate(th3, kernel, iterations=1)

        # plot
        '''
        titles2 = ['Original', 'Adaptive', "Otsu", 'Gaussian + Otsu']
        images2 = [img, dilation, dilation2, dilation3]
        for i in range(4):
            plt.subplot(2, 2, i + 1), plt.imshow(images2[i], 'gray')
            plt.title(titles2[i])
            plt.xticks([]), plt.yticks([])
        plt.show()
        '''

        erosion = cv2.erode(dilation, kernel, iterations=1)
        erosion2 = cv2.erode(dilation2, kernel, iterations=1)
        erosion3 = cv2.erode(dilation3, kernel, iterations=1)

        '''
        # plot
        titles3 = ['Original', 'Adaptive', "Otsu", 'Gaussian + Otsu']
        images3 = [img, erosion, erosion2, erosion3]
        for i in range(4):
            plt.subplot(2, 2, i + 1), plt.imshow(images3[i], 'gray')
            plt.title(titles3[i])
            plt.xticks([]), plt.yticks([])
        plt.show()
        '''

        #############

        # Now we perform a last Morphological Transformation but this time the kernel is 3x1
        # to reduce the height of the line
        kernel = np.ones((3, 1), np.uint8)
        dilation = cv2.dilate(erosion, kernel, iterations=1)
        dilation2 = cv2.dilate(erosion2, kernel, iterations=1)
        dilation3 = cv2.dilate(erosion3, kernel, iterations=1)

        '''
        # plot
        titles4 = ['Original', 'Adaptive', "Otsu", 'Gaussian + Otsu']
        images4 = [img, dilation, dilation2, dilation3]
        for i in range(4):
            plt.subplot(2, 2, i + 1), plt.imshow(images4[i], 'gray')
            plt.title(titles4[i])
            plt.xticks([]), plt.yticks([])
        plt.show()
        '''

        #############

        # There is no way to isolate the letters (by removing the line) without distoring
        # the letters and making them irecognizable. Since the letter are always at the
        # same place, we can simply extract them by hardcoding the coordinates.

        # contouring
        # get the individual letters
        x, y, w, h = 30, 12, 20, 38
        contours = []
        for i in range(5):
            # get the bounding rect
            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.rectangle(dilation, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.rectangle(dilation2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #cv2.rectangle(dilation3, (x, y), (x + w, y + h), (0, 255, 0), 2)

            '''
            roi = img[y:y + h, x: x + w]
            cv2.imwrite("{}/{}/{}.png".format(args["output"], "original", i), roi)
        
            roi = dilation[y:y + h, x: x + w]
            cv2.imwrite("{}/{}/{}.png".format(args["output"], "adaptive", i), roi)
        
            roi = dilation2[y:y + h, x: x + w]
            cv2.imwrite("{}/{}/{}.png".format(args["output"], "otsu", i), roi)
        
            roi = dilation3[y:y + h, x: x + w]
            cv2.imwrite("{}/{}/{}.png".format(args["output"], "gaussian_otsu", i), roi)
            x += w
            '''
            roi = dilation3[y:y + h, x: x + w]
            contours.append(roi)
            x += w
        '''
        # plot
        titles5 = ['Original', 'Adaptive', "Otsu", 'Gaussian + Otsu']
        images5 = [img, dilation, dilation2, dilation3]
        for i in range(4):
            plt.subplot(2, 2, i + 1), plt.imshow(images5[i], 'gray')
            plt.title(titles5[i])
            plt.xticks([]), plt.yticks([])
        plt.show()
        '''
        return contours

    @staticmethod
    def preprocess(image, width, height):
        # grab the dimensions of the image, then initialize
        # the padding values
        (h, w) = image.shape[:2]
        # if the width is greater than the height then resize along
        # the width
        if w > h:
            image = imutils.resize(image, width=width)
        # otherwise, the height is greater than the width so resize
        # along the height
        else:
            image = imutils.resize(image, height=height)

        # determine the padding values for the width and height to
        # obtain the target dimensions
        padW = int((width - image.shape[1]) / 2.0)
        padH = int((height - image.shape[0]) / 2.0)

        # pad the image then apply one more resizing to handle any
        # rounding issues
        image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
        image = cv2.resize(image, (width, height))

        # return the pre-processed image
        return image
