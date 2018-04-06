# import the necessary packages
# import the necessary packages
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras import backend as K
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
    # "class_1": "../berries/under/",
    "class_2": "../berries/ok/",
}

# # get H5 files ready for writing
# h5_under = tables.openFile('under.h5', mode='a', title="Under_H5")
# u_root = h5_under.root
h5_ok = tables.openFile('ok.h5', mode='a', title="OK_H5")
ok_root = h5_ok.root

#batch size
bs = 16

#functions class
fun = Functions()

#image to look for
imageType = 'RGBTop'

#get numer of images
imagePaths = fun.getImagePaths(args, imageType)

# load the ResNet50 network (i.e., the network we'll be using for
# feature extraction)
model = ResNet50(weights="imagenet", include_top=False)

#progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ",
    progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# initialize our data matrix (where we will store our extracted
# features)
data = None
count = 0


for path in imagePaths:
    #get image and preprocess
    image = fun.preProcessImage(path)
    features = model.predict(image)
    h5_ok.createArray(ok_root, "under_{}".format(count), features)
    count+=1
    #save vectors, labals
    # with open("Vectors.pkl", "wb") as tf:
    #     pickle.dump("Vector: {}, Label: {}\r\n".format(features[0], 'Underripe'), tf)
    # update the progress bar
    pbar.update(count)


# finish up the progress bar
pbar.finish()


# h5_under.close()
h5_ok.close()
