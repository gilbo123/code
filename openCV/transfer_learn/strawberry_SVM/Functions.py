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
    Gets the image paths and
    returns them
    '''
    def getImagePaths(self, paths, imageType):
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
    def preProcessImage(self, path):
        #get image
        img = plt.imread(path, 0)
        #crop
        cropped = img[0:1200, 600:1800]
        #trans to HSV, saturation channel
        #have to invert colours first
        destRGB = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        # blurred = cv2.medianBlur(destRGB, 15)
        # hsv = cv2.cvtColor(destRGB, cv2.COLOR_BGR2HSV)
        # h, s, v = cv2.split(hsv)
        # #threshhold saturation channel
        # (T, threshInv) = cv2.threshold(s, 185, 255, cv2.THRESH_BINARY)
        # # thresh = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4)
        # #apply mask
        # masked = cv2.bitwise_and(destRGB, destRGB, mask = threshInv)
        # small = cv2.resize(blurred, (0,0), fx=0.5, fy=0.5)
        # cv2.imshow('Image-thresh', small)
        # cv2.waitKey(0)
        # resize to ResNet image input size
        res = cv2.resize(destRGB, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        #flatten
        image = img_to_array(res)

        # preprocess the image by (1) expanding the dimensions and
        # (2) subtracting the mean RGB pixel intensity from the
        # ImageNet dataset
        image = np.expand_dims(image, axis=0)
        # image = imagenet_utils.preprocess_input(image)
        return image
