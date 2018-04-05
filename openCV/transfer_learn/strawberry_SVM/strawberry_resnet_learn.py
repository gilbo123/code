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
    "class_1": "../berries/under/",
    "class_2": "../berries/ok/",
}

print(len(args))

#functions class
fun = Functions()

imagetypeRGB = 'RGBTop'

# load the ResNet50 network (i.e., the network we'll be using for
# feature extraction)
model = ResNet50(weights="imagenet", include_top=False)

batchImages = []
classLabels = []

for i in range(len(args)):
    print(arg[i])


# subdirs = [x[0] for x in os.walk(args["class_1"])]
# for subdir in subdirs:
#     files = os.walk(subdir).next()[2]
#     if (len(files) > 0):
#         for file in files:
#             if (imagetypeRGB in file):
#                 ###
#                 imagePath = os.path.join(subdir, file)
#                 print imagePath
#
#                 #get image
#                 #image = load_img(imagePath, target_size=(224, 224))
#                 cropped = plt.imread(imagePath)
#                 cropped = cropped[0:1200, 600:1800]
#                 # resized = K.resize_images(cropped, 224, 224, 'channels_first')
#                 res = cv2.resize(cropped, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
#                 #flatten
#                 image = img_to_array(res)
#
#                 # preprocess the image by (1) expanding the dimensions and
#                 # (2) subtracting the mean RGB pixel intensity from the
#                 # ImageNet dataset
#                 image = np.expand_dims(image, axis=0)
#                 image = imagenet_utils.preprocess_input(image)
#
#                 # add the image to the batch
#                 batchImages.append(image)
#                 #add a label
#                 classLabels.append("Class-1")
#
#
# #second class images
# subdirs = [x[0] for x in os.walk(args["class_2"])]
# for subdir in subdirs:
#     files = os.walk(subdir).next()[2]
#     if (len(files) > 0):
#         for file in files:
#             if (imagetypeRGB in file):
#                 imagePath = os.path.join(subdir, file)
#                 print imagePath
#
#                 # load the input image using the Keras helper utility
#                 # while ensuring the image is resized to 224x224 pixels
#                 #image = load_img(imagePath, target_size=(224, 224))
#                 cropped = plt.imread(imagePath)
#                 cropped = cropped[0:1200, 600:1800]
#                 # resized = K.resize_images(cropped, 224, 224, 'channels_first')
#                 res = cv2.resize(cropped, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
#                 #flatten
#                 image = img_to_array(res)
#
#                 # preprocess the image by (1) expanding the dimensions and
#                 # (2) subtracting the mean RGB pixel intensity from the
#                 # ImageNet dataset
#                 image = np.expand_dims(image, axis=0)
#                 image = imagenet_utils.preprocess_input(image)
#
#                 # add the image to the batch
#                 batchImages.append(image)
#                 #add a label
#                 classLabels.append("Class-2")
#
#
#
# # pass the images through the network and use the outputs as
# # our actual features
# batchImages = np.vstack(batchImages)
# features = model.predict(batchImages)
# # reshape the features so that each image is represented by
# # a flattened feature vector of the `MaxPooling2D` outputs
# print features.shape
# features = features.reshape((features.shape[0], 2048))
# # show the data matrix shape and amount of memory it consumes
# print(features.shape)
# print(features.nbytes)
#
# #make sure type is correct
# trainingDataMat = np.array(features, np.float32)
#
# #encode labels
# le = LabelEncoder()
# labels = le.fit_transform(classLabels)
#
# # define the set of parameters that we want to tune then start a
# # grid search where we evaluate our model for each value of C
# print("[INFO] adding to SVM...")
# svm = cv2.ml.SVM_create()
# svm.setKernel(cv2.ml.SVM_LINEAR)
# svm.setType(cv2.ml.SVM_C_SVC)
# svm.setC(5.67)
# #svm.setGamma(5.383)
#
# svm.train(trainingDataMat, cv2.ml.ROW_SAMPLE, labels)
# svm.save('svm_data.dat')
