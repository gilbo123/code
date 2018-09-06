from matplotlib import pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import os
import cv2


class Functions:
    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals
        self.members = ['Tiger', 'Elephant', 'Wild Cat']

    def printMembers(self):
        print('Printing members of the Mammals class')
        for member in self.members:
            print('\t%s ' % member)

    '''
    Gets the image paths
    from current folder and
    returns them
    '''
    def getImagePaths(self, paths):
        #set counter
        imagePaths = []
        #go through all folders
        for name, path in paths.iteritems():
            #print(name)
            files = os.walk(path).next()[2]
            if (len(files) > 0):
                for file in files:
                    #print(file)
                    imagePath = os.path.join(path, file)
                    imagePaths.append(imagePath)

        return imagePaths



    '''
    Gets the image paths
    from a structured folder
    set and returns them
    '''
    def getStructuredImagePaths(self, paths, imageType):
        #set counter
        imagePaths = []
        #go through all folders
        for name, path in paths.iteritems():
            #print(name)
            subdirs = [x[0] for x in os.walk(path)]
            for subdir in subdirs:
                files = os.walk(subdir).next()[2]
                if (len(files) > 0):
                    for file in files:
                        if (imageType in file):
                            imagePath = os.path.join(subdir, file)
                            imagePaths.append(imagePath)

        return imagePaths


    '''
    Pre-Processing
    '''
    def preProcessImage(self, img, crop=False):

        if crop:
            #crop if required
            img = img[0:1200, 600:1800]
        #trans to HSV, saturation channel
        #have to invert colours first
        destRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # blurred = cv2.medianBlur(destRGB, 25)
        # hsv = cv2.cvtColor(destRGB, cv2.COLOR_BGR2HSV)
        # h, s, v = cv2.split(hsv)
        # #threshhold saturation channel
        # (T, threshInv1) = cv2.threshold(s, 180, 255, cv2.THRESH_BINARY)
        # # thresh = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4)
        # #apply mask
        # sat = cv2.bitwise_and(destRGB, destRGB, mask = threshInv1)
        # (T, threshInv2) = cv2.threshold(s, 180, 255, cv2.THRESH_BINARY)
        # white = cv2.bitwise_and(destRGB, destRGB, mask = threshInv2)
        # masked = cv2.bitwise_or(sat, white)
        # #convert to LAB colourspace
        # Lab = cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)
        # L, a, b = cv2.split(Lab)
        # #find any light patches
        # #subtract b from a
        # sub = a-b
        # (T, threshInv) = cv2.threshold(sub, 200, 255, cv2.THRESH_BINARY)
        # #apply mask
        # masked2 = cv2.bitwise_and(destRGB, destRGB, mask = threshInv)

        #small = cv2.resize(masked, (0,0), fx=0.5, fy=0.5)
        # resize to ResNet image input size
        res = cv2.resize(destRGB, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)

        # cv2.imshow("image", masked)
        # cv2.imshow("image2", res)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #flatten
        image = img_to_array(res)

        # preprocess the image by (1) expanding the dimensions and
        # (2) subtracting the mean RGB pixel intensity from the
        # ImageNet dataset
        image = np.expand_dims(image, axis=0)
        # image = imagenet_utils.preprocess_input(image)
        return image


    '''
    Get the colour values
    of each pixel in the frame
    '''
    def getColourVals(self, path, crop=False):
        #get image
        img = plt.imread(path, 0)
        if crop:
            #crop if required
            img = img[0:1200, 600:1800]

        #reverse RGB
        destRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #blur image
        blurred = cv2.medianBlur(destRGB, 25)

        # resize to ResNet image input size
        res = cv2.resize(blurred, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

        pixels = []
        #go though each pixel to find RGB values
        for p in res:
            pix = [p[0], p[1], p[2]]
            pixels = np.append(pixels, pix)

        return pixels
