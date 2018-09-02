# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import load_img
#from keras.preprocessing.image import apply_transform
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, MaxPooling1D, GlobalAveragePooling1D, MaxPooling2D, Convolution2D, Conv2DTranspose
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from sklearn.model_selection import GridSearchCV
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
bs = 8

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

#ResNet Output vector length
out_vector = 18432

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
    features = features.reshape((features.shape[0], out_vector))
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


'''
Classify vectors
'''
'''
C = 0.1

#Decision tree
decisiontree = DecisionTreeClassifier()
decisiontree.fit(data[:i], labels[:i])
y_pred = decisiontree.predict(data[i:])
acc_decisiontree = round(accuracy_score(y_pred, labels[i:]) * 100, 2)
print('Decision Tree classifier: {}'.format(acc_decisiontree))


rfc = RandomForestClassifier()
rfc.fit(data[:i], labels[:i])
y_pred = rfc.predict(data[i:])
acc_rfc = round(accuracy_score(y_pred, labels[i:]) * 100, 2)
print('Random Forest classifier: {}'.format(acc_rfc))


adaBoost = AdaBoostClassifier()
adaBoost.fit(data[:i], labels[:i])
y_pred = adaBoost.predict(data[i:])
acc_adaBoost = round(accuracy_score(y_pred, labels[i:]) * 100, 2)
print('AdaBoost classifier: {}'.format(acc_adaBoost))



svc = LinearSVC(C=C)
svc.fit(data[:i], labels[:i])
y_pred = svc.predict(data[i:])
acc_svc = round(accuracy_score(y_pred, labels[i:]) * 100, 2)
print('Linear SVC classifier: {}'.format(acc_svc))


log = LogisticRegression(C=C)
log.fit(data[:i], labels[:i])
y_pred = log.predict(data[i:])
acc_log = round(accuracy_score(y_pred, labels[i:]) * 100, 2)
print('Logistic Regression classifier: {}'.format(acc_log))




gauss = GaussianNB()
gauss.fit(data[:i], labels[:i])
y_pred = gauss.predict(data[i:])
acc_g = round(accuracy_score(y_pred, labels[i:]) * 100, 2)
print('Gaussian classifier: {}'.format(acc_g))




per = Perceptron()
per.fit(data[:i], labels[:i])
y_pred = per.predict(data[i:])
acc_per = round(accuracy_score(y_pred, labels[i:]) * 100, 2)
print('Perceptron classifier: {}'.format(acc_per))




kn = KNeighborsClassifier()
kn.fit(data[:i], labels[:i])
y_pred = kn.predict(data[i:])
acc_kn = round(accuracy_score(y_pred, labels[i:]) * 100, 2)
print('K-Neighbors classifier: {}'.format(acc_kn))


mlp = MLPClassifier(solver='sgd', learning_rate='adaptive')
mlp.fit(data[:i], labels[:i])
y_pred = mlp.predict(data[i:])
acc_kn = round(accuracy_score(y_pred, labels[i:]) * 100, 2)
print('MLP classifier: {}'.format(acc_kn))
'''


data = np.expand_dims(data, axis=2)
print(data.shape)
model = Sequential()
model.add(Conv1D(16, 3, activation='relu', input_shape=(out_vector, 1)))
model.add(Conv1D(16, 3, activation='relu'))
model.add(MaxPooling1D(3))
#model.add(Conv1D(128, 3, activation='relu'))
#model.add(Conv1D(128, 3, activation='relu'))
#model.add(GlobalAveragePooling1D())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])


model.fit(data[:i], labels[:i], batch_size=16, epochs=10)
#score = model.evaluate(y_pred, labels[i:], batch_size=16)

y_pred = model.predict(data[i:])
acc_nn = round(accuracy_score(y_pred, labels[i:]) * 100, 2)
print('Neural Net classifier: {}'.format(acc_nn))


#save model
#joblib.dump(model, "resnet_model.pkl")
