# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

from timeit import default_timer as timer
from imutils import paths
from Functions import Functions
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
    "class_1": "../berries/foreign/",
    "class_2": "../berries/ok/",
}

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
classLabels = []

# loop over the images in batches
for i in np.arange(0, len(imagePaths), bs):
    # extract the batch of images and labels, then initialize the
    # list of actual images that will be passed through the network
    # for feature extraction
    batchPaths = imagePaths[i:i + bs]
    batchImages = []
    batchLabels = []

    for path in batchPaths:
        #get image and preprocess
        image = fun.preProcessImage(path)
        # add the image to the batch
        batchImages.append(image)
        #add a label
        label = "/".join(path.split("/")[2:3])
        classLabels.append(label)

    # pass the images through the network and use the outputs as
    # our actual features
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)
    # reshape the features so that each image is represented by
    # a flattened feature vector of the `MaxPooling2D` outputs
    features = features.reshape((features.shape[0], 2048))
    # show the data matrix shape and amount of memory it consumes
    # print(features.shape)
    # print(features.nbytes)

    # if our data matrix is None, initialize it
    if data is None:
        data = features

    # otherwise, stack the data and features together
    else:
        data = np.vstack([data, features])

    # update the progress bar
    pbar.update(i)


# finish up the progress bar
pbar.finish()

#make sure type is correct
trainingDataMat = np.array(data, np.float32)

#encode labels
le = LabelEncoder()
labels = le.fit_transform(classLabels)
# print(labels)

# define the set of parameters that we want to tune then start a
# grid search where we evaluate our model for each value of C
print("[INFO] adding to SVM...")
svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_RBF)
svm.setType(cv2.ml.SVM_C_SVC)
# svm.setC(5.67)
#svm.setGamma(5.383)

svm.train(trainingDataMat, cv2.ml.ROW_SAMPLE, labels)
svm.save('svm_data.dat')
