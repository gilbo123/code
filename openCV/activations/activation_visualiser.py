# import the necessary packages
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from scipy.spatial import distance
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

from timeit import default_timer as timer
from matplotlib import pyplot as plt
from imutils import paths
from Functions import Functions
import tables
import numpy as np
import progressbar
import random
import os
import cv2

#from this website
#https://blogs.technet.microsoft.com/machinelearning/2018/02/22/22-minutes-to-2nd-place-in-a-kaggle-competition-with-deep-learning-azure/


# since we are not using command line arguments (like we typically
# would inside Deep Learning for Computer Vision with Python, let's
# "pretend" we are by using an `args` dictionary -- this will enable
# us to easily reuse and swap out code depending if we are using the
# command line or Jupyter Notebook
args = {
    "dataset": "test_images/",
}


#functions class
fun = Functions()

#image to look for
imageType = 'RGBTop'

#get numer of images
imagePaths = fun.getImagePaths(args, imageType)

# load the ResNet50 network (i.e., the network we'll be using for
# feature extraction)
model = ResNet50(weights="imagenet", include_top=False)

for path in imagePaths:
    # load the input image using the Keras helper utility
    image = fun.preProcessImage(path)

    #c_img, m_img = fun.getGreyFromColour(path)

    #get features from network
    features_c = model.predict(image)
    #features_m = model.predict(m_img)

    # reshape the features so that each image is represented by
    # a flattened feature vector of the `MaxPooling2D` outputs
    features_c = features_c.reshape((features_c.shape[0], 2048))#8192
    #features_m = features_m.reshape((features_m.shape[0], 2048))
    #difference = features_c - features_m

    #numpy format 32-bit
    features = np.array(features_c, np.float32)

    # get image and plot
    fig = plt.figure()
    img = plt.imread(path, 0)

    # fig.add_subplot(211)
    # plt.title('image')
    # plt.imshow(img)

    # fig.add_subplot(211)
    plt.title('activations ')
    plt.plot(features_c[0])

    # fig.add_subplot(212)
    # plt.title('activations ')
    # plt.plot(features_m[0])

    plt.show()
    cv2.waitKey(0)

    # blurred = np.hstack([cv2.medianBlur(image, 3),
    #                     cv2.medianBlur(image, 5),
    #                     cv2.medianBlur(image, 7)])
    # cv2.imshow("Median", blurred)
    # cv2.waitKey(0)
