# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import apply_transform
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from matplotlib import pyplot as plt

from timeit import default_timer as timer
from imutils import paths
from Functions import Functions
import numpy as np
import progressbar
import random
import os
import cv2

import pickle




#from this website
#https://blogs.technet.microsoft.com/machinelearning/2018/02/22/22-minutes-to-2nd-place-in-a-kaggle-competition-with-deep-learning-azure/


# since we are not using command line arguments (like we typically
# would inside Deep Learning for Computer Vision with Python, let's
# "pretend" we are by using an `args` dictionary -- this will enable
# us to easily reuse and swap out code depending if we are using the
# command line or Jupyter Notebook
args = {
    "class_1": "../berries/under/",
    "class_2": "../berries/pass/",
}

#batch size
bs = 16

counter = 0

#functions class
fun = Functions()

#image to look for
imageType = 'RGBTop'

#dimensions
(h, w) = 600, 500
# calculate the center of the image
center = (w / 2, h / 2)
#angles to rotate
angle90 = 90
angle180 = 180
angle270 = 270
#rotation scale
scale = 1.0

#get numer of images
imagePaths = fun.getImagePaths(args)
random.shuffle(imagePaths)
print(len(imagePaths))

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
        img = plt.imread(path, 0)
        #img = load_img(path, target_size=(224, 224))
        #normal image
        p_img = fun.preProcessImage(img)

        #add a label
        label = "/".join(path.split("/")[2:3])

        classLabels.append(label)
        batchImages.append(p_img)

        # cv2.imshow("vertical flip", horizontal_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

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


# determine the index of the training and testing split (75% for
# training and 25% for testing)
i = int(data.shape[0] * 0.75)

# print(data[:i])
# print(labels[:i])
# print("#########################")
# print(data[i:])
# print(labels[i:])

# define the set of parameters that we want to tune then start a
# grid search where we evaluate our model for each value of C
# print("[INFO] tuning hyperparameters...")
# params = {"C": [0.0001, 0.001, 0.01, 0.1, 1.0]}
# clf = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=-1)
# clf.fit(data[:i], labels[:i])
# print("[INFO] best hyperparameters: {}".format(clf.best_params_))
#
# # generate a classification report for the model
# print("[INFO] evaluating...")
# preds = clf.predict(data[i:])
# print(classification_report(labels[i:], preds, target_names=le.classes_))
#
# # compute the raw accuracy with extra precision
# acc = accuracy_score(labels[i:], preds)
# print("[INFO] score: {}".format(acc))

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score)
    clf.fit(data[:i], labels[:i])

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = labels[i:], clf.predict(data[i:])
    print(classification_report(y_true, y_pred))
    print()

pickle.dump(clf, open("resnet_model_new.pkl", 'w'))

#define the set of parameters that we want to tune then start a
#grid search where we evaluate our model for each value of C
# print("[INFO] adding to SVM...")
# svm = cv2.ml.SVM_create()
# svm.setKernel(cv2.ml.SVM_RBF)
# svm.setType(cv2.ml.SVM_C_SVC)
# # svm.setC(5.67)
# #svm.setGamma(5.383)
#
# svm.train(trainingDataMat, cv2.ml.ROW_SAMPLE, labels)
# svm.save('svm_data.dat')
